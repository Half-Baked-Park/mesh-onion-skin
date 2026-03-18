bl_info = {
    "name": "Mesh Onion Skin",
    "author": "HB PARK",
    "version": (1, 3, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Onion Skin",
    "description": "GPU-based onion skin ghosts for 3D mesh animations",
    "category": "Animation",
}

import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, IntProperty, FloatProperty,
    FloatVectorProperty, EnumProperty, PointerProperty,
)
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
_pending_rebuild = None  # (scene,)
# Preserve original show_in_front value before mesh_in_front is applied {object_name: bool}
_original_mesh_show_in_front: dict[str, bool] = {}

_PERF_WARN_THRESHOLD = 10


def _get_shader():
    return gpu.shader.from_builtin('UNIFORM_COLOR')


# ---------------------------------------------------------------------------
# Target collection
# ---------------------------------------------------------------------------

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


def _has_animation(obj) -> bool:
    """Check if the object has any animation data."""
    if obj.animation_data and obj.animation_data.action:
        return True
    arm = _find_armature(obj)
    if arm and _get_active_action(arm):
        return True
    if obj.data and hasattr(obj.data, 'shape_keys') and obj.data.shape_keys:
        if obj.data.shape_keys.animation_data:
            return True
    return False


def _collect_target_meshes(scene=None, context=None) -> list:
    """Return list of target mesh objects based on mode."""
    if scene is None:
        try:
            scene = context.scene if context else bpy.context.scene
        except AttributeError:
            return []
    try:
        props = scene.mesh_onion_skin
    except AttributeError:
        return []

    if props.mode == 'ACTIVE':
        ctx = context if context is not None else bpy.context
        obj = _get_target_mesh(ctx)
        return [obj] if obj else []

    # SCENE / COLLECTION mode
    if props.mode == 'COLLECTION':
        col = props.target_collection
        if col is None:
            return []
        source = col.all_objects
    else:  # SCENE
        source = scene.collection.all_objects

    candidates = [o for o in source if o.type == 'MESH']

    # Filters: visible + animated only
    candidates = [o for o in candidates if o.visible_get()]
    candidates = [o for o in candidates if _has_animation(o)]

    # Limit to max objects
    max_obj = props.max_objects
    if len(candidates) > max_obj:
        candidates = candidates[:max_obj]

    return candidates



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

    # Keyframe mode only works in Active mode
    if props.use_keyframes and props.mode == 'ACTIVE':
        _status, keyframes = _get_armature_keyframes(obj)
        before = [f for f in keyframes if f < current]
        after  = [f for f in keyframes if f > current]
        if props.count_before > 0:
            frames.extend(before[-props.count_before:])
        if props.count_after > 0:
            frames.extend(after[:props.count_after])
    else:
        step = props.frame_step
        for i in range(1, props.count_before + 1):
            frames.append(current - i * step)
        for i in range(1, props.count_after + 1):
            frames.append(current + i * step)

    return frames


# ---------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------

def _bake_mesh_snapshot(obj, depsgraph, use_flat: bool):
    """Return a GPU batch snapshot of the mesh with depsgraph already set."""
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


def rebuild_cache(scene, targets=None):
    """Incrementally build the onion skin cache for target objects."""
    global _is_baking
    if _is_baking:
        return

    props = scene.mesh_onion_skin
    if not props.enabled:
        return

    if targets is None:
        targets = _collect_target_meshes()
    if not targets:
        clear_cache()
        return

    # Evict stale cache entries
    valid_names = {obj.name for obj in targets}
    stale = [k for k in _onion_cache if k not in valid_names]
    for k in stale:
        _onion_cache.pop(k, None)

    # Collect per-object target frames + build frame-first baking map
    obj_target_frames: dict[str, set[int]] = {}
    frames_to_objects: dict[int, list] = {}
    for obj in targets:
        frame_list = _get_target_frames(scene, props, obj)
        if not frame_list:
            clear_cache(obj.name)
            continue
        target_set = set(frame_list)
        obj_target_frames[obj.name] = target_set

        # Compare with existing cache, collect only frames that need baking
        existing = _onion_cache.get(obj.name, {})
        new_cache: dict[int, gpu.types.GPUBatch] = {}
        for f in frame_list:
            if f in existing:
                new_cache[f] = existing[f]
            else:
                frames_to_objects.setdefault(f, []).append(obj)
        _onion_cache[obj.name] = new_cache

    if not frames_to_objects:
        return

    # Frame-first loop: minimize frame_set calls
    current = scene.frame_current
    _is_baking = True
    try:
        for frame in sorted(frames_to_objects.keys()):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for obj in frames_to_objects[frame]:
                batch = _bake_mesh_snapshot(obj, depsgraph, props.use_flat)
                if batch is not None:
                    _onion_cache.setdefault(obj.name, {})[frame] = batch
    finally:
        try:
            scene.frame_set(current)
        except Exception:
            pass
        _is_baking = False


# ---------------------------------------------------------------------------
# GPU draw
# ---------------------------------------------------------------------------

def draw_onion_skins():
    """Viewport draw callback — renders cached ghost meshes."""
    scene = bpy.context.scene
    props = scene.mesh_onion_skin
    if not props.enabled:
        return

    # Draw only objects already in cache (rebuild_cache handles collection/baking)
    targets = [bpy.data.objects.get(n) for n in _onion_cache if bpy.data.objects.get(n)]
    if not targets:
        return

    current = scene.frame_current
    shader = _get_shader()

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_mask_set(False)
    if props.in_front == 'GHOST':
        gpu.state.depth_test_set('NONE')
    else:
        gpu.state.depth_test_set('LESS_EQUAL')
    if props.use_flat:
        gpu.state.line_width_set(1.5)

    shader.bind()

    def _draw_group(frames, color_rgb, cache):
        n = len(frames)
        for i, frame in enumerate(frames):
            batch = cache.get(frame)
            if batch is None:
                continue
            if props.use_fade:
                t = (i + 1) / (n + 1)
                factor = (1.0 - t) ** props.fade_falloff
                alpha = props.opacity * factor
            else:
                alpha = props.opacity
            shader.uniform_float("color", (*color_rgb[:3], alpha))
            batch.draw(shader)

    for obj in targets:
        cache = _onion_cache.get(obj.name)
        if not cache:
            continue
        before_sorted = sorted([f for f in cache if f < current], reverse=True)
        after_sorted  = sorted([f for f in cache if f > current])
        _draw_group(before_sorted, props.color_before, cache)
        _draw_group(after_sorted,  props.color_after, cache)

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
    try:
        props = scene.mesh_onion_skin
    except AttributeError:
        return
    if not props.enabled:
        return
    targets = _collect_target_meshes(scene=scene)
    rebuild_cache(scene, targets)
    # Request viewport redraw after cache update
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


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
    scene, targets = data
    try:
        props = scene.mesh_onion_skin
    except AttributeError:
        return None
    if not props.enabled:
        return None
    rebuild_cache(scene, targets)
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
    clear_cache()
    try:
        scene = context.scene if context else bpy.context.scene
    except AttributeError:
        return
    # Capture targets now while context is valid
    targets = _collect_target_meshes(scene=scene, context=context)
    _pending_rebuild = (scene, targets)
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
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.mode')
    _schedule_rebuild(context)
    _tag_redraw(context)


def _update_mode(self, context):
    """On mode switch — clear cache and rebuild."""
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


def _update_in_front(self, context):
    """On in-front mode change — apply/restore mesh show_in_front."""
    targets = _collect_target_meshes(context=context)
    for obj in targets:
        if self.in_front == 'MESH':
            if obj.name not in _original_mesh_show_in_front:
                _original_mesh_show_in_front[obj.name] = obj.show_in_front
            obj.show_in_front = True
        else:
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
    mode: EnumProperty(
        name="Mode",
        items=[
            ('ACTIVE', "Active", "Show ghosts for the active object only", 'OBJECT_DATA', 0),
            ('SCENE', "Scene", "Show ghosts for all mesh objects in the scene", 'SCENE_DATA', 1),
            ('COLLECTION', "Collection", "Show ghosts for all mesh objects in a collection", 'OUTLINER_COLLECTION', 2),
        ],
        default='ACTIVE',
        update=_update_mode,
    )
    target_collection: PointerProperty(
        type=bpy.types.Collection,
        name="Collection",
        description="Target collection for onion skin",
        update=_update_cache,
    )
    max_objects: IntProperty(
        name="Max Objects", default=10, min=1, max=50,
        description="Maximum number of objects to process in Scene/Collection mode",
        update=_update_cache,
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
        description="Frame step interval",
        update=_update_cache,
    )
    use_keyframes: BoolProperty(
        name="Keyframes Only", default=False,
        description="Show ghosts only at armature keyframe positions (Active mode only)",
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
    in_front: EnumProperty(
        name="In Front",
        items=[
            ('NONE', "None", "Use default depth testing"),
            ('GHOST', "Ghost", "Always draw ghosts in front"),
            ('MESH', "Mesh", "Always draw the mesh object in front"),
        ],
        default='GHOST',
        update=_update_in_front,
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
        props.enabled = not props.enabled
        return {'FINISHED'}


class MESH_OT_onion_skin_update(Operator):
    bl_idname = "mesh.onion_skin_update"
    bl_label = "Update Onion Skin"
    bl_description = "Force rebuild the onion skin cache"

    def execute(self, context):
        clear_cache()
        targets = _collect_target_meshes(context=context)
        rebuild_cache(context.scene, targets)
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

        # Mode selector
        layout.prop(props, "mode", text="")

        # Target settings (Scene/Collection modes only)
        if props.mode != 'ACTIVE':
            box = layout.box()
            box.label(text="Targets")
            if props.mode == 'COLLECTION':
                box.prop(props, "target_collection", text="")
            box.prop(props, "max_objects")
            targets = _collect_target_meshes(context=context)
            count = len(targets)
            box.label(
                text=f"  {count} objects",
                icon='MESH_DATA',
            )
            if count >= props.max_objects:
                box.label(
                    text="  Max objects reached",
                    icon='ERROR',
                )

        # Frame settings
        box = layout.box()
        box.label(text="Frames")
        row = box.row(align=True)
        row.prop(props, "count_before")
        row.prop(props, "count_after")

        if props.mode == 'ACTIVE':
            # Keyframes: only shown in Active mode
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
        else:
            box.prop(props, "frame_step")

        # Display settings
        box = layout.box()
        box.label(text="Display")
        box.prop(props, "opacity", slider=True)
        box.prop(props, "use_fade")
        sub = box.row()
        sub.active = props.use_fade
        sub.prop(props, "fade_falloff", slider=True)
        box.prop(props, "in_front")
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
