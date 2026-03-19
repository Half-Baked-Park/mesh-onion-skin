bl_info = {
    "name": "Mesh Onion Skin",
    "author": "HB PARK",
    "version": (2, 0, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Onion Skin",
    "description": "GPU-based onion skin ghosts for 3D mesh animations",
    "category": "Animation",
}

import bpy
import gpu
import numpy as np
import functools
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

# {object_name: {frame_number: (positions, indices)}}
_onion_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
_draw_handle = None
_is_baking = False
_rebuild_scheduled = False
_pending_rebuild = None  # (scene,)
# Preserve original show_in_front value before mesh_in_front is applied {object_name: bool}
_original_mesh_show_in_front: dict[str, bool] = {}

_PERF_WARN_THRESHOLD = 10

# --- Progressive baking state ---
_bake_queue: list[tuple[int, list]] = []  # (frame, [objects]) 우선순위 큐
_bake_generation: int = 0  # 세대 카운터 — 스크러빙 시 stale 작업 취소
_bake_timer_running: bool = False
_bake_progress: float = 0.0  # 0.0~1.0 베이킹 진행률
_bake_total_frames: int = 0  # 현재 베이킹 작업의 총 프레임 수

# --- Merged batch state (Phase 2) ---
_merged_before: gpu.types.GPUBatch | None = None
_merged_after: gpu.types.GPUBatch | None = None
_merged_dirty: bool = True


def _get_shader():
    return gpu.shader.from_builtin('SMOOTH_COLOR')


# ---------------------------------------------------------------------------
# Frustum culling (Phase 3)
# ---------------------------------------------------------------------------

def _get_active_3d_view():
    """Return (region, region_3d) for the active 3D viewport, or (None, None)."""
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return region, area.spaces.active.region_3d
    except Exception:
        pass
    return None, None


def _extract_frustum_planes(perspective_matrix) -> np.ndarray:
    """Extract 6 frustum planes from VP matrix (Gribb-Hartmann method). Returns (6, 4) array."""
    m = np.array(perspective_matrix, dtype=np.float32)
    planes = np.empty((6, 4), dtype=np.float32)
    planes[0] = m[3] + m[0]   # Left
    planes[1] = m[3] - m[0]   # Right
    planes[2] = m[3] + m[1]   # Bottom
    planes[3] = m[3] - m[1]   # Top
    planes[4] = m[3] + m[2]   # Near
    planes[5] = m[3] - m[2]   # Far
    # Normalize
    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    planes /= norms
    return planes


def _is_in_frustum(obj, frustum_planes: np.ndarray) -> bool:
    """Test if object's bounding box intersects the view frustum."""
    bb = obj.bound_box  # 8 corners in local space
    mat = np.array(obj.matrix_world, dtype=np.float32)
    corners_h = np.empty((8, 4), dtype=np.float32)
    for i, c in enumerate(bb):
        corners_h[i, 0] = c[0]
        corners_h[i, 1] = c[1]
        corners_h[i, 2] = c[2]
    corners_h[:, 3] = 1.0
    world = (corners_h @ mat.T)[:, :3]  # (8, 3)
    # If all 8 corners are outside any single plane → outside frustum
    for i in range(6):
        dots = world @ frustum_planes[i, :3] + frustum_planes[i, 3]
        if np.all(dots < 0):
            return False
    return True


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
    """Remove geometry cache and invalidate merged batches."""
    global _merged_before, _merged_after, _merged_dirty
    if obj_name:
        _onion_cache.pop(obj_name, None)
    else:
        _onion_cache.clear()
    _merged_before = None
    _merged_after = None
    _merged_dirty = True


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
    """Return (positions, indices) numpy arrays for the mesh snapshot."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    if mesh is None or len(mesh.vertices) == 0:
        eval_obj.to_mesh_clear()
        return None

    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)

    # World-space transform — pre-allocated homogeneous coordinates (avoids hstack crash)
    mat = np.array(eval_obj.matrix_world, dtype=np.float32)
    co_h = np.empty((n, 4), dtype=np.float32)
    co_h[:, :3] = co
    co_h[:, 3] = 1.0
    co = np.ascontiguousarray((co_h @ mat.T)[:, :3])

    if use_flat:
        edge_n = len(mesh.edges)
        if edge_n == 0:
            eval_obj.to_mesh_clear()
            return None
        idx = np.empty(edge_n * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", idx)
        idx = idx.reshape(-1, 2)
    else:
        mesh.calc_loop_triangles()
        tri_n = len(mesh.loop_triangles)
        if tri_n == 0:
            eval_obj.to_mesh_clear()
            return None
        idx = np.empty(tri_n * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", idx)
        idx = idx.reshape(-1, 3)

    eval_obj.to_mesh_clear()
    return (co, idx)


def _build_prioritized_queue(current_frame: int, frames_to_objects: dict[int, list]) -> list[tuple[int, list]]:
    """Sort frames by proximity to current frame. Closest frames bake first."""
    items = list(frames_to_objects.items())
    items.sort(key=lambda pair: abs(pair[0] - current_frame))
    return items


def rebuild_cache(scene, targets=None, force_clear: bool = False):
    """Compute delta and enqueue progressive baking. Non-blocking."""
    global _bake_generation, _bake_timer_running, _bake_progress, _bake_total_frames, _merged_dirty
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

    # Full clear when format changes (e.g. wireframe toggle)
    if force_clear:
        clear_cache()

    # Evict stale cache entries
    valid_names = {obj.name for obj in targets}
    stale = [k for k in _onion_cache if k not in valid_names]
    for k in stale:
        _onion_cache.pop(k, None)

    # Collect per-object target frames + build frame-first baking map (delta only)
    frames_to_objects: dict[int, list] = {}
    for obj in targets:
        frame_list = _get_target_frames(scene, props, obj)
        if not frame_list:
            clear_cache(obj.name)
            continue

        # Preserve existing valid cache entries, evict frames no longer needed
        existing = _onion_cache.get(obj.name, {})
        new_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for f in frame_list:
            if f in existing:
                new_cache[f] = existing[f]
            else:
                frames_to_objects.setdefault(f, []).append(obj)
        _onion_cache[obj.name] = new_cache

    _merged_dirty = True

    if not frames_to_objects:
        return

    # Cancel any in-progress bake
    _bake_generation += 1

    # Build priority queue — closest frames first
    _bake_queue.clear()
    _bake_queue.extend(_build_prioritized_queue(scene.frame_current, frames_to_objects))
    _bake_total_frames = len(_bake_queue)
    _bake_progress = 0.0

    # Start progressive bake timer
    # Always register a new timer — old one will self-abort on generation mismatch
    _bake_timer_running = True
    gen = _bake_generation
    bpy.app.timers.register(
        functools.partial(_progressive_bake_tick, gen),
        first_interval=0.0,
    )


def _progressive_bake_tick(generation: int) -> float | None:
    """Timer callback — bakes N frames per tick, yields back to Blender."""
    global _is_baking, _bake_timer_running, _bake_progress, _merged_dirty

    # Stale generation — abort
    if generation != _bake_generation:
        _bake_timer_running = False
        return None

    # Nothing left — done
    if not _bake_queue:
        _bake_timer_running = False
        _bake_progress = 1.0
        return None

    try:
        scene = bpy.context.scene
        props = scene.mesh_onion_skin
    except (AttributeError, RuntimeError):
        _bake_timer_running = False
        return None

    if not props.enabled:
        _bake_queue.clear()
        _bake_timer_running = False
        return None

    # Determine batch size (frames per tick)
    batch_size = max(1, getattr(props, 'bake_batch_size', 2))
    current = scene.frame_current
    _is_baking = True

    try:
        frames_done = 0
        while _bake_queue and frames_done < batch_size:
            # Re-check generation inside loop
            if generation != _bake_generation:
                _bake_timer_running = False
                return None

            frame, objects = _bake_queue.pop(0)
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for obj_ref in objects:
                # Re-fetch object to avoid stale Python references
                obj = bpy.data.objects.get(obj_ref.name)
                if obj is None:
                    continue
                try:
                    geo = _bake_mesh_snapshot(obj, depsgraph, props.use_flat)
                except Exception:
                    continue
                if geo is not None:
                    _onion_cache.setdefault(obj.name, {})[frame] = geo
                    _merged_dirty = True
            frames_done += 1
    finally:
        # Restore current frame to prevent viewport flicker
        try:
            scene.frame_set(current)
        except Exception:
            pass
        _is_baking = False

    # Update progress
    if _bake_total_frames > 0:
        done = _bake_total_frames - len(_bake_queue)
        _bake_progress = done / _bake_total_frames

    # Request viewport redraw to show newly baked ghosts
    _request_viewport_redraw()

    if _bake_queue:
        return 0.0  # re-schedule immediately for next tick
    else:
        _bake_timer_running = False
        _bake_progress = 1.0
        return None


def _request_viewport_redraw():
    """Request redraw for all 3D viewports."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GPU draw — merged batch system
# ---------------------------------------------------------------------------

def _build_merged_batches():
    """Merge all cached ghost geometry into 2 mega-batches (before + after current frame)."""
    global _merged_before, _merged_after, _merged_dirty
    _merged_before = None
    _merged_after = None
    _merged_dirty = False

    if not _onion_cache:
        return

    try:
        scene = bpy.context.scene
        props = scene.mesh_onion_skin
    except (AttributeError, RuntimeError):
        return

    current = scene.frame_current
    use_flat = props.use_flat
    prim_type = 'LINES' if use_flat else 'TRIS'

    # Collect geometry per group
    before_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    after_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    before_offset = 0
    after_offset = 0

    # Frustum culling — skip objects outside viewport
    frustum_planes = None
    if getattr(props, 'use_frustum_cull', True) and props.mode != 'ACTIVE':
        _region, rv3d = _get_active_3d_view()
        if rv3d:
            try:
                frustum_planes = _extract_frustum_planes(rv3d.perspective_matrix)
            except Exception:
                pass

    for obj_name, cache in _onion_cache.items():
        if not cache:
            continue

        # Frustum cull — skip objects outside camera view
        if frustum_planes is not None:
            obj = bpy.data.objects.get(obj_name)
            if obj and not _is_in_frustum(obj, frustum_planes):
                continue

        before_frames = sorted([f for f in cache if f < current], reverse=True)
        after_frames = sorted([f for f in cache if f > current])

        # Before ghosts
        n_before = len(before_frames)
        for i, frame in enumerate(before_frames):
            geo = cache.get(frame)
            if geo is None:
                continue
            pos, idx = geo
            n_verts = len(pos)
            if props.use_fade:
                t = (i + 1) / (n_before + 1)
                alpha = props.opacity * ((1.0 - t) ** props.fade_falloff)
            else:
                alpha = props.opacity
            vc = np.empty((n_verts, 4), dtype=np.float32)
            vc[:, 0] = props.color_before[0]
            vc[:, 1] = props.color_before[1]
            vc[:, 2] = props.color_before[2]
            vc[:, 3] = alpha
            before_parts.append((pos, vc, idx + before_offset))
            before_offset += n_verts

        # After ghosts
        n_after = len(after_frames)
        for i, frame in enumerate(after_frames):
            geo = cache.get(frame)
            if geo is None:
                continue
            pos, idx = geo
            n_verts = len(pos)
            if props.use_fade:
                t = (i + 1) / (n_after + 1)
                alpha = props.opacity * ((1.0 - t) ** props.fade_falloff)
            else:
                alpha = props.opacity
            vc = np.empty((n_verts, 4), dtype=np.float32)
            vc[:, 0] = props.color_after[0]
            vc[:, 1] = props.color_after[1]
            vc[:, 2] = props.color_after[2]
            vc[:, 3] = alpha
            after_parts.append((pos, vc, idx + after_offset))
            after_offset += n_verts

    shader = _get_shader()

    if before_parts:
        m_pos = np.concatenate([p[0] for p in before_parts])
        m_col = np.concatenate([p[1] for p in before_parts])
        m_idx = np.concatenate([p[2] for p in before_parts])
        _merged_before = batch_for_shader(
            shader, prim_type,
            {"pos": m_pos, "color": m_col},
            indices=m_idx.tolist(),
        )

    if after_parts:
        m_pos = np.concatenate([p[0] for p in after_parts])
        m_col = np.concatenate([p[1] for p in after_parts])
        m_idx = np.concatenate([p[2] for p in after_parts])
        _merged_after = batch_for_shader(
            shader, prim_type,
            {"pos": m_pos, "color": m_col},
            indices=m_idx.tolist(),
        )


def draw_onion_skins():
    """Viewport draw callback — renders 2 merged batches (before + after)."""
    global _merged_dirty
    scene = bpy.context.scene
    props = scene.mesh_onion_skin
    if not props.enabled:
        return
    if not _onion_cache:
        return

    # Rebuild merged batches if geometry or display settings changed
    if _merged_dirty:
        _build_merged_batches()

    if _merged_before is None and _merged_after is None:
        return

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

    if _merged_before is not None:
        _merged_before.draw(shader)
    if _merged_after is not None:
        _merged_after.draw(shader)

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
    global _merged_dirty
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
    _merged_dirty = True  # Alpha values depend on current frame
    _request_viewport_redraw()


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
    scene, targets, force_clear = data
    try:
        props = scene.mesh_onion_skin
    except AttributeError:
        return None
    if not props.enabled:
        return None
    rebuild_cache(scene, targets, force_clear=force_clear)
    _request_viewport_redraw()
    return None


def _schedule_rebuild(context=None, force_clear: bool = False):
    """Schedule a rebuild on the next timer tick. Does NOT clear cache by default."""
    global _rebuild_scheduled, _pending_rebuild
    try:
        scene = context.scene if context else bpy.context.scene
    except AttributeError:
        return
    # Capture targets now while context is valid
    targets = _collect_target_meshes(scene=scene, context=context)
    _pending_rebuild = (scene, targets, force_clear)
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
    """Incremental rebuild — reuses overlapping cached frames."""
    scene = context.scene if context else bpy.context.scene
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.use_keyframes')
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.use_flat')
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.mode')
    _schedule_rebuild(context)
    _tag_redraw(context)


def _update_cache_full(self, context):
    """Full rebuild — clears all cache (for format changes like wireframe toggle)."""
    scene = context.scene if context else bpy.context.scene
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.use_keyframes')
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.use_flat')
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.mode')
    _schedule_rebuild(context, force_clear=True)
    _tag_redraw(context)


def _update_mode(self, context):
    """On mode switch — full clear and rebuild (target set changes completely)."""
    _schedule_rebuild(context, force_clear=True)
    _tag_redraw(context)


def _update_enabled(self, context):
    """On enable toggle — works for both the header checkbox and operator button."""
    if self.enabled:
        _schedule_rebuild(context, force_clear=True)
    else:
        clear_cache()
    _tag_redraw(context)


def _update_display(self, context):
    """For settings that only need a redraw (color, opacity, fade change)."""
    global _merged_dirty
    _merged_dirty = True  # Colors/opacity baked into per-vertex data
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
        name="Max Objects", default=10, min=1, max=500,
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
        update=_update_cache_full,
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
        update=_update_cache_full,
    )
    bake_batch_size: IntProperty(
        name="Bake Batch", default=2, min=1, max=10,
        description="Frames to bake per timer tick (higher = faster bake, more stutter)",
    )
    use_frustum_cull: BoolProperty(
        name="Frustum Cull", default=True,
        description="Skip drawing ghosts for objects outside the camera view",
        update=_update_display,
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
        targets = _collect_target_meshes(context=context)
        rebuild_cache(context.scene, targets, force_clear=True)
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

        # Performance settings (Scene/Collection modes only)
        if props.mode != 'ACTIVE':
            box = layout.box()
            box.label(text="Performance")
            box.prop(props, "bake_batch_size")
            box.prop(props, "use_frustum_cull")

        # Bake progress indicator
        if _bake_timer_running:
            box = layout.box()
            box.label(
                text=f"Baking... {_bake_progress:.0%}",
                icon='SORTTIME',
            )

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
    global _draw_handle, _rebuild_scheduled, _bake_timer_running
    # Cancel progressive bake timer
    if _bake_timer_running:
        _bake_queue.clear()
        _bake_timer_running = False
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
