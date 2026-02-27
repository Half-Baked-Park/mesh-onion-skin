bl_info = {
    "name": "Mesh Onion Skin",
    "author": "Claude",
    "version": (1, 2, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Onion Skin",
    "description": "GPU-based onion skin ghosts for 3D mesh animations",
    "category": "Animation",
}

import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, IntProperty, FloatProperty, FloatVectorProperty
from bpy.types import PropertyGroup, Operator, Panel
from gpu_extras.batch import batch_for_shader


# ---------------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------------

# {object_name: {frame_number: GPUBatch}}
_onion_cache: dict[str, dict[int, gpu.types.GPUBatch]] = {}
_draw_handle = None
_is_baking = False
_rebuild_scheduled = False
_pending_rebuild = None  # (scene, obj_name)
# Preserve original show_in_front value before mesh_in_front is applied {object_name: bool}
_original_mesh_show_in_front: dict[str, bool] = {}


def _get_shader():
    return gpu.shader.from_builtin('UNIFORM_COLOR')


def _get_target_mesh(context=None):
    """Return the active object or the mesh child of an armature in pose mode."""
    ctx = context if context is not None else bpy.context
    obj = ctx.view_layer.objects.active
    if obj is None:
        return None
    if obj.type == 'MESH':
        return obj
    if obj.type == 'ARMATURE':
        for child in obj.children:
            if child.type == 'MESH':
                return child
    return None


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def clear_cache(obj_name: str | None = None):
    """Remove GPU batch cache."""
    if obj_name:
        _onion_cache.pop(obj_name, None)
    else:
        _onion_cache.clear()


def _find_armature(obj):
    """Return the armature linked to the object (parent first, then modifier)."""
    if obj.parent and obj.parent.type == 'ARMATURE':
        return obj.parent
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object:
            return mod.object
    return None


def _get_active_action(arm):
    """Return the current active action of the armature (direct → NLA tweak → first NLA strip)."""
    ad = getattr(arm, 'animation_data', None)
    if ad is None:
        return None
    if ad.action:
        return ad.action
    if ad.nla_tracks:
        for track in ad.nla_tracks:
            for strip in track.strips:
                if strip.active and strip.action:
                    return strip.action
        for track in ad.nla_tracks:
            for strip in track.strips:
                if strip.action:
                    return strip.action
    return None


def _collect_keyframes_from_action(action) -> set[int]:
    """Collect keyframe numbers from action (Blender 5.0 Layered Action + legacy fallback)."""
    kf_set: set[int] = set()
    # Blender 5.0+ Layered Action: action.layers → strips → channelbags → fcurves
    try:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        for kp in fc.keyframe_points:
                            kf_set.add(round(kp.co[0]))
    except (AttributeError, TypeError):
        pass
    # Legacy fallback: direct action.fcurves access
    if not kf_set:
        try:
            for fc in action.fcurves:
                for kp in fc.keyframe_points:
                    kf_set.add(round(kp.co[0]))
        except (AttributeError, RuntimeError):
            pass
    return kf_set


def _get_armature_keyframes(obj) -> tuple[str, list[int]]:
    """Collect keyframes from the armature's current action. Returns (status string, frame list)."""
    arm = _find_armature(obj)
    if arm is None:
        return "No armature", []
    action = _get_active_action(arm)
    if action is None:
        return f"{arm.name}: No active action", []
    kf_set = _collect_keyframes_from_action(action)
    if kf_set:
        return f"{arm.name} > {action.name}: {len(kf_set)} keys", sorted(kf_set)
    return f"{arm.name} > {action.name}: 0 keys", []


def _get_target_frames(scene, props, obj) -> list[int]:
    """Return the list of frame numbers where ghosts should be displayed."""
    current = scene.frame_current
    frames: list[int] = []

    if props.use_keyframes:
        _status, keyframes = _get_armature_keyframes(obj)
        # Split keyframes into before/after relative to current
        before = [f for f in keyframes if f < current]
        after  = [f for f in keyframes if f > current]
        # Explicit check because before[-0:] would return the entire list when count=0
        if props.count_before > 0:
            frames.extend(before[-props.count_before:])
        if props.count_after > 0:
            frames.extend(after[:props.count_after])
    else:
        step = props.frame_step
        for i in range(1, props.count_before + 1):
            f = current - i * step
            if f >= scene.frame_start:
                frames.append(f)
        for i in range(1, props.count_after + 1):
            f = current + i * step
            if f <= scene.frame_end:
                frames.append(f)

    return frames


def _bake_frame(scene, obj, frame: int, use_flat: bool):
    """Jump to the given frame and return a GPU batch snapshot of the mesh."""
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    if mesh is None or len(mesh.vertices) == 0:
        eval_obj.to_mesh_clear()
        return None

    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)

    mat = np.array(eval_obj.matrix_world, dtype=np.float32)
    ones = np.ones((n, 1), dtype=np.float32)
    co = np.ascontiguousarray((np.hstack((co, ones)) @ mat.T)[:, :3])

    shader = _get_shader()

    if use_flat:
        edge_n = len(mesh.edges)
        if edge_n == 0:
            eval_obj.to_mesh_clear()
            return None
        idx = np.empty(edge_n * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", idx)
        batch = batch_for_shader(
            shader, 'LINES', {"pos": co},
            indices=idx.reshape(-1, 2).tolist())
    else:
        mesh.calc_loop_triangles()
        tri_n = len(mesh.loop_triangles)
        if tri_n == 0:
            eval_obj.to_mesh_clear()
            return None
        idx = np.empty(tri_n * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", idx)
        batch = batch_for_shader(
            shader, 'TRIS', {"pos": co},
            indices=idx.reshape(-1, 3).tolist())

    eval_obj.to_mesh_clear()
    return batch


def rebuild_cache(scene, obj=None):
    """Incrementally build the onion skin cache for the active object."""
    global _is_baking
    if _is_baking:
        return

    props = scene.mesh_onion_skin
    if not props.enabled:
        return

    if obj is None:
        obj = _get_target_mesh()
    if obj is None:
        return

    name = obj.name
    targets = _get_target_frames(scene, props, obj)
    if not targets:
        clear_cache(name)
        return

    existing = _onion_cache.get(name, {})
    target_set = set(targets)
    if set(existing.keys()) == target_set:
        return

    new_cache: dict[int, gpu.types.GPUBatch] = {}
    to_bake: list[int] = []
    for f in targets:
        if f in existing:
            new_cache[f] = existing[f]
        else:
            to_bake.append(f)

    if not to_bake:
        _onion_cache[name] = new_cache
        return

    current = scene.frame_current
    _is_baking = True
    try:
        for f in to_bake:
            batch = _bake_frame(scene, obj, f, props.use_flat)
            if batch is not None:
                new_cache[f] = batch
    finally:
        try:
            scene.frame_set(current)
        except Exception:
            pass
        _is_baking = False

    _onion_cache[name] = new_cache



# ---------------------------------------------------------------------------
# GPU draw
# ---------------------------------------------------------------------------

def draw_onion_skins():
    """Viewport draw callback — renders cached ghost meshes."""
    scene = bpy.context.scene
    props = scene.mesh_onion_skin
    if not props.enabled:
        return

    obj = _get_target_mesh()
    if obj is None:
        return

    cache = _onion_cache.get(obj.name)
    if not cache:
        return

    current = scene.frame_current
    shader = _get_shader()

    # Sort before/after frames closest-first (for index-based fade)
    before_sorted = sorted([f for f in cache if f < current], reverse=True)  # closest first
    after_sorted  = sorted([f for f in cache if f > current])                 # closest first

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_mask_set(False)
    if props.ghost_in_front:
        gpu.state.depth_test_set('NONE')
    else:
        gpu.state.depth_test_set('LESS_EQUAL')
    if props.use_flat:
        gpu.state.line_width_set(1.5)

    shader.bind()

    def _draw_group(frames, color_rgb):
        n = len(frames)
        for i, frame in enumerate(frames):
            batch = cache.get(frame)
            if batch is None:
                continue
            if props.use_fade:
                # Index-based fade: closest (i=0) → full opacity
                #                   farthest (i=n-1) → 0
                t = i / max(n - 1, 1)
                factor = (1.0 - t) ** props.fade_falloff
                alpha = props.opacity * factor
            else:
                alpha = props.opacity
            shader.uniform_float("color", (*color_rgb[:3], alpha))
            batch.draw(shader)

    _draw_group(before_sorted, props.color_before)
    _draw_group(after_sorted,  props.color_after)

    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(True)
    if props.use_flat:
        gpu.state.line_width_set(1.0)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@persistent
def _on_frame_change(scene, depsgraph):
    if _is_baking:
        return
    props = scene.mesh_onion_skin
    if not props.enabled:
        return
    rebuild_cache(scene)


@persistent
def _on_load_post(*_args):
    clear_cache()


# ---------------------------------------------------------------------------
# Timer-based rebuild
# ---------------------------------------------------------------------------

def _do_rebuild():
    global _rebuild_scheduled, _pending_rebuild
    _rebuild_scheduled = False
    data = _pending_rebuild
    _pending_rebuild = None
    if data is None:
        return None
    scene, obj_name = data
    try:
        props = scene.mesh_onion_skin
    except AttributeError:
        return None
    if not props.enabled:
        return None
    # Use bpy.data.objects instead of bpy.context.view_layer → safe in timer context
    obj = bpy.data.objects.get(obj_name) if obj_name else None
    rebuild_cache(scene, obj)
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass
    return None


def _schedule_rebuild(context=None):
    global _rebuild_scheduled, _pending_rebuild
    obj = _get_target_mesh(context)
    obj_name = obj.name if obj else None
    clear_cache(obj_name)
    try:
        scene = context.scene if context else bpy.context.scene
    except AttributeError:
        return
    _pending_rebuild = (scene, obj_name)
    if not _rebuild_scheduled:
        _rebuild_scheduled = True
        bpy.app.timers.register(_do_rebuild, first_interval=0.0)


# ---------------------------------------------------------------------------
# Property update callbacks
# ---------------------------------------------------------------------------

def _tag_redraw(context):
    if context and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _clear_fcurve_if_present(scene, data_path: str):
    """Remove the fcurve for the given path from the scene action. Prevents overwriting during frame_set()."""
    ad = scene.animation_data
    if not ad or not ad.action:
        return
    # Blender 5.0+ Layered Action
    try:
        for layer in ad.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    fc = bag.fcurves.find(data_path)
                    if fc:
                        bag.fcurves.remove(fc)
    except AttributeError:
        pass
    # Legacy fallback
    try:
        fc = ad.action.fcurves.find(data_path)
        if fc:
            ad.action.fcurves.remove(fc)
    except (AttributeError, RuntimeError):
        pass


def _update_cache(self, context):
    # Prevent keyframe data from overwriting values during frame_set(): remove relevant fcurves
    scene = context.scene if context else bpy.context.scene
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.use_keyframes')
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.use_flat')
    _schedule_rebuild(context)
    _tag_redraw(context)


def _update_enabled(self, context):
    """On enable toggle — works for both the header checkbox and operator button."""
    if self.enabled:
        _schedule_rebuild(context)
    else:
        clear_cache()
    _tag_redraw(context)


def _update_display(self, context):
    """For settings that only need a redraw."""
    _tag_redraw(context)


def _update_mesh_in_front(self, context):
    """On mesh-in-front toggle — apply to the actual object property."""
    obj = _get_target_mesh(context)
    if obj:
        if self.mesh_in_front:
            # Preserve original value only if not already saved
            if obj.name not in _original_mesh_show_in_front:
                _original_mesh_show_in_front[obj.name] = obj.show_in_front
            obj.show_in_front = True
        else:
            # Restore original value (default False if not saved)
            obj.show_in_front = _original_mesh_show_in_front.pop(obj.name, False)
    _tag_redraw(context)


# ---------------------------------------------------------------------------
# Property group
# ---------------------------------------------------------------------------

class MeshOnionSkinProps(PropertyGroup):
    enabled: BoolProperty(
        name="Enabled",
        description="Show onion skin",
        default=False,
        update=_update_enabled,
    )
    count_before: IntProperty(
        name="Before", default=3, min=0, max=10,
        description="Number of ghosts before the current frame",
        update=_update_cache,
    )
    count_after: IntProperty(
        name="After", default=3, min=0, max=10,
        description="Number of ghosts after the current frame",
        update=_update_cache,
    )
    frame_step: IntProperty(
        name="Step", default=1, min=1, max=10,
        description="Frame step interval (used when keyframe mode is off)",
        update=_update_cache,
    )
    use_keyframes: BoolProperty(
        name="Keyframes Only", default=False,
        description="Show ghosts only at armature keyframe positions",
        update=_update_cache,
    )
    color_before: FloatVectorProperty(
        name="Before Color", subtype='COLOR_GAMMA',
        size=3, default=(0.2, 0.8, 0.2), min=0.0, max=1.0,
        description="Ghost color for frames before current",
        update=_update_display,
    )
    color_after: FloatVectorProperty(
        name="After Color", subtype='COLOR_GAMMA',
        size=3, default=(0.2, 0.4, 0.9), min=0.0, max=1.0,
        description="Ghost color for frames after current",
        update=_update_display,
    )
    opacity: FloatProperty(
        name="Opacity", default=0.5, min=0.0, max=1.0,
        subtype='FACTOR',
        description="Ghost opacity",
        update=_update_display,
    )
    use_fade: BoolProperty(
        name="Fade", default=True,
        description="Decrease opacity with distance",
        update=_update_display,
    )
    fade_falloff: FloatProperty(
        name="Fade Falloff", default=1.0, min=0.2, max=5.0,
        subtype='FACTOR',
        description="Fade curve strength (higher = greater difference between near and far ghosts)",
        update=_update_display,
    )
    ghost_in_front: BoolProperty(
        name="Ghost In Front", default=False,
        description="Always draw onion skin ghosts in front of the mesh",
        update=_update_display,
    )
    mesh_in_front: BoolProperty(
        name="Mesh In Front", default=False,
        description="Always draw the mesh object in front (object show_in_front property)",
        update=_update_mesh_in_front,
    )
    use_flat: BoolProperty(
        name="Wireframe", default=False,
        description="Display ghosts as wireframe",
        update=_update_cache,
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class MESH_OT_onion_skin_toggle(Operator):
    bl_idname = "mesh.onion_skin_toggle"
    bl_label = "Toggle Onion Skin"
    bl_description = "Toggle onion skin display"

    def execute(self, context):
        props = context.scene.mesh_onion_skin
        props.enabled = not props.enabled  # _update_enabled callback handles the rest
        return {'FINISHED'}


class MESH_OT_onion_skin_update(Operator):
    bl_idname = "mesh.onion_skin_update"
    bl_label = "Update Onion Skin"
    bl_description = "Force rebuild the onion skin cache"

    def execute(self, context):
        obj = _get_target_mesh(context)
        if obj:
            clear_cache(obj.name)
        else:
            clear_cache()
        rebuild_cache(context.scene)
        _tag_redraw(context)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class MESH_PT_onion_skin(Panel):
    bl_label = "Onion Skin"
    bl_idname = "MESH_PT_onion_skin"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Onion Skin"

    def draw_header(self, context):
        props = context.scene.mesh_onion_skin
        self.layout.prop(props, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        props = context.scene.mesh_onion_skin
        layout.active = props.enabled

        # Frame settings
        box = layout.box()
        box.label(text="Frames")
        row = box.row(align=True)
        row.prop(props, "count_before")
        row.prop(props, "count_after")
        box.prop(props, "use_keyframes")
        sub = box.row()
        sub.active = not props.use_keyframes
        sub.prop(props, "frame_step")
        if props.use_keyframes:
            obj = _get_target_mesh(context)
            if obj:
                status, _kfs = _get_armature_keyframes(obj)
                has_kfs = len(_kfs) > 0
                box.label(
                    text=f"  {status}",
                    icon='ARMATURE_DATA' if has_kfs else 'ERROR',
                )

        # Display settings
        box = layout.box()
        box.label(text="Display")
        box.prop(props, "opacity", slider=True)
        box.prop(props, "use_fade")
        sub = box.row()
        sub.active = props.use_fade
        sub.prop(props, "fade_falloff", slider=True)
        box.prop(props, "ghost_in_front")
        box.prop(props, "mesh_in_front")
        box.prop(props, "use_flat")

        # Color settings
        box = layout.box()
        box.label(text="Colors")
        row = box.row(align=True)
        row.prop(props, "color_before", text="")
        row.prop(props, "color_after", text="")

        # Action buttons
        row = layout.row(align=True)
        toggle_text = "Disable" if props.enabled else "Enable"
        toggle_icon = 'PAUSE' if props.enabled else 'PLAY'
        row.operator("mesh.onion_skin_toggle", text=toggle_text,
                     icon=toggle_icon, depress=props.enabled)
        row.operator("mesh.onion_skin_update", text="", icon='FILE_REFRESH')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    MeshOnionSkinProps,
    MESH_OT_onion_skin_toggle,
    MESH_OT_onion_skin_update,
    MESH_PT_onion_skin,
)


def register():
    global _draw_handle
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_onion_skin = bpy.props.PointerProperty(
        type=MeshOnionSkinProps)
    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    bpy.app.handlers.load_post.append(_on_load_post)
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        draw_onion_skins, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _draw_handle, _rebuild_scheduled
    if _rebuild_scheduled:
        try:
            bpy.app.timers.unregister(_do_rebuild)
        except (ValueError, RuntimeError):
            pass
        _rebuild_scheduled = False
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    # Restore show_in_front values changed by mesh_in_front
    for obj_name, original in _original_mesh_show_in_front.items():
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.show_in_front = original
    _original_mesh_show_in_front.clear()
    clear_cache()
    bpy.app.handlers.load_post.remove(_on_load_post)
    bpy.app.handlers.frame_change_post.remove(_on_frame_change)
    del bpy.types.Scene.mesh_onion_skin
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
