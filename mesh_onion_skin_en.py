bl_info = {
    "name": "Mesh Onion Skin",
    "author": "HB PARK",
    "version": (2, 3, 1),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Onion Skin",
    "description": "GPU-based onion skin ghosts for 3D mesh animations",
    "category": "Animation",
}

import bpy
import gpu
import numpy as np
from functools import partial
from collections import deque
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, IntProperty, FloatProperty,
    FloatVectorProperty, EnumProperty, PointerProperty,
)
from bpy.types import PropertyGroup, Operator, Panel
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix


# ---------------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------------

# {object_name: {frame_number: (positions, indices)}}
_onion_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
# {armature_name: (action_name, sorted_keyframe_frames)} — avoids re-walking keyframes every frame
_keyframe_cache: dict[str, tuple[str, list[int]]] = {}
_draw_handle = None
_is_baking = False
_rebuild_scheduled = False
_pending_rebuild = None  # (scene,)

# --- Progressive baking state ---
_bake_queue: deque[tuple[int, list]] = deque()  # (frame, [obj_names]) 우선순위 큐
_bake_generation: int = 0  # 세대 카운터 — 스크러빙 시 stale 작업 취소
_bake_timer_running: bool = False
_bake_progress: float = 0.0  # 0.0~1.0 베이킹 진행률
_bake_total_frames: int = 0  # 현재 베이킹 작업의 총 프레임 수

# --- Merged batch state (Phase 2) ---
_merged_before: gpu.types.GPUBatch | None = None
_merged_after: gpu.types.GPUBatch | None = None
_merged_dirty: bool = True

# --- Mesh-in-front (MESH) occluder: stamps current pose at near-plane depth to hide ghosts ---
# Unlike show_in_front, works in all shading modes (solid / material / rendered)
_occluder_batch: gpu.types.GPUBatch | None = None
_building_occluder: bool = False  # re-entrancy guard for the depsgraph handler

# --- Debounced ghost rebuild after an edit (keyframe re-timing / pose editing) ---
_EDIT_SETTLE: float = 0.2  # rebuild once edits have been quiet this long (avoids frame_set mid-drag)
_edit_seq: int = 0         # bumped on every detected edit
_edit_ack: int = 0         # last value the settle timer observed
_edit_rebuild_armed: bool = False

# --- Same-pose detection (current frame snapshots) ---
_current_frame_snapshots: dict[str, np.ndarray] = {}


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
    corners_h[:, :3] = np.array(bb, dtype=np.float32)
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
    """Check if the object has any animation source (action, drivers, NLA, constraints, or animated parent)."""
    ad = obj.animation_data
    if ad:
        if ad.action:
            return True
        if ad.drivers:
            return True
        if ad.nla_tracks:
            return True
    arm = _find_armature(obj)
    if arm and _get_active_action(arm):
        return True
    if obj.data and hasattr(obj.data, 'shape_keys') and obj.data.shape_keys:
        if obj.data.shape_keys.animation_data:
            return True
    if obj.constraints:
        return True
    # Child of an animated parent (e.g. parented to animated Empty/Armature)
    if obj.parent and obj.parent.type != 'ARMATURE':
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
    global _merged_before, _merged_after, _merged_dirty, _occluder_batch
    if obj_name:
        _onion_cache.pop(obj_name, None)
    else:
        _onion_cache.clear()
        _keyframe_cache.clear()
    _merged_before = None
    _merged_after = None
    _merged_dirty = True
    _occluder_batch = None


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


def _fcurve_key_frames(fc, kf_set: set[int]) -> None:
    """Bulk-read one fcurve's keyframe frame numbers into kf_set (C-speed foreach_get)."""
    n = len(fc.keyframe_points)
    if n == 0:
        return
    co = np.empty(n * 2, dtype=np.float32)
    fc.keyframe_points.foreach_get("co", co)
    # co = [frame0, value0, frame1, value1, ...] → take frame components, round to int
    kf_set.update(np.rint(co[0::2]).astype(np.int64).tolist())


def _collect_keyframes_from_action(action) -> set[int]:
    """Collect keyframe numbers from action (Blender 5.0 Layered Action + legacy fallback)."""
    kf_set: set[int] = set()
    # Blender 5.0+ Layered Action: action.layers → strips → channelbags → fcurves
    try:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        _fcurve_key_frames(fc, kf_set)
    except (AttributeError, TypeError):
        pass
    # Legacy fallback: direct action.fcurves access
    if not kf_set:
        try:
            for fc in action.fcurves:
                _fcurve_key_frames(fc, kf_set)
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
    # Cache per armature — re-collect only when the action changes or the cache is cleared.
    # (After editing keyframes in the same action, press the Update button to refresh.)
    cached = _keyframe_cache.get(arm.name)
    if cached is not None and cached[0] == action.name:
        frames = cached[1]
    else:
        frames = sorted(_collect_keyframes_from_action(action))
        _keyframe_cache[arm.name] = (action.name, frames)
    if frames:
        return f"{arm.name} > {action.name}: {len(frames)} keys", frames
    return f"{arm.name} > {action.name}: 0 keys", []


def _get_target_frames(scene, props, obj) -> list[int]:
    """Return the list of frame numbers where ghosts should be displayed."""
    current = scene.frame_current
    frames: list[int] = []

    if props.use_keyframes:
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

def _bake_mesh_snapshot(obj, depsgraph, use_flat: bool, ghost_detail: float = 1.0):
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

    # Ghost Detail — reduce triangle/edge count by uniform sampling
    if ghost_detail < 1.0 and len(idx) > 1:
        keep = max(1, int(len(idx) * ghost_detail))
        step = max(1, len(idx) // keep)
        idx = idx[::step][:keep]

    return (co, idx)


def _build_prioritized_queue(current_frame: int, frames_to_objects: dict[int, list]) -> list[tuple[int, list]]:
    """Sort frames by proximity to current frame. Closest frames bake first."""
    items = list(frames_to_objects.items())
    items.sort(key=lambda pair: abs(pair[0] - current_frame))
    return items


def _bake_queue_item(scene, props, frame, obj_names) -> None:
    """Bake one (frame, obj_names) queue entry into the cache. Shared by sync + progressive paths.

    Caller is responsible for setting _is_baking and restoring the current frame.
    """
    global _merged_dirty
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj_name in obj_names:
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        try:
            geo = _bake_mesh_snapshot(obj, depsgraph, props.use_flat, props.ghost_detail)
        except Exception:
            continue
        if geo is None:
            continue
        # Skip ghost if pose is identical to current frame
        if props.skip_same_pose:
            cur_snap = _current_frame_snapshots.get(obj_name)
            if cur_snap is not None and cur_snap.shape == geo[0].shape:
                if np.allclose(cur_snap, geo[0], atol=1e-4):
                    continue
        _onion_cache.setdefault(obj.name, {})[frame] = geo
        _merged_dirty = True


def _bake_all_sync(scene, props) -> None:
    """Bake the whole queue immediately (blocking). Used when sync_bake (live follow) is on."""
    global _is_baking
    current = scene.frame_current
    _is_baking = True
    try:
        while _bake_queue:
            frame, obj_names = _bake_queue.popleft()
            _bake_queue_item(scene, props, frame, obj_names)
    finally:
        # Restore current frame to prevent viewport flicker
        try:
            scene.frame_set(current)
        except Exception:
            pass
        _is_baking = False


def rebuild_cache(scene, targets=None, force_clear: bool = False):
    """Compute delta and enqueue progressive baking. Non-blocking."""
    global _bake_generation, _bake_timer_running, _bake_progress, _bake_total_frames, _merged_dirty
    if _is_baking:
        return

    props = scene.mesh_onion_skin
    if not props.enabled:
        return

    if targets is None:
        targets = _collect_target_meshes(scene=scene)
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

    # Rebuild current-pose occluder (MESH mode only, for near-plane depth)
    _build_occluder(scene, props, targets)

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
                frames_to_objects.setdefault(f, []).append(obj.name)
        _onion_cache[obj.name] = new_cache

    _merged_dirty = True

    if not frames_to_objects:
        return

    # Cancel any in-progress bake
    _bake_generation += 1

    # Capture current-frame snapshots for same-pose detection (once per rebuild)
    _current_frame_snapshots.clear()
    if props.skip_same_pose:
        depsgraph_cur = bpy.context.evaluated_depsgraph_get()
        all_obj_names = {n for names in frames_to_objects.values() for n in names}
        for obj_name in all_obj_names:
            obj = bpy.data.objects.get(obj_name)
            if obj is None:
                continue
            try:
                geo = _bake_mesh_snapshot(obj, depsgraph_cur, props.use_flat, props.ghost_detail)
                if geo is not None:
                    _current_frame_snapshots[obj_name] = geo[0]
            except Exception:
                pass

    # Build priority queue — closest frames first
    _bake_queue.clear()
    _bake_queue.extend(_build_prioritized_queue(scene.frame_current, frames_to_objects))
    _bake_total_frames = len(_bake_queue)
    _bake_progress = 0.0

    # Sync (live-follow) bake — bake everything now so ghosts follow during scrub/playback
    if props.sync_bake:
        _bake_all_sync(scene, props)
        _bake_timer_running = False
        _bake_progress = 1.0
        _merged_dirty = True
        return

    # Start progressive bake timer
    # Always register a new timer — old one will self-abort on generation mismatch
    _bake_timer_running = True
    gen = _bake_generation
    bpy.app.timers.register(
        partial(_progressive_bake_tick, gen),
        first_interval=0.0,
    )


def _progressive_bake_tick(generation: int) -> float | None:
    """Timer callback — bakes N frames per tick, yields back to Blender."""
    global _is_baking, _bake_timer_running, _bake_progress

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
    batch_size = max(1, props.bake_batch_size)
    current = scene.frame_current
    _is_baking = True

    try:
        frames_done = 0
        while _bake_queue and frames_done < batch_size:
            # Re-check generation inside loop
            if generation != _bake_generation:
                _bake_timer_running = False
                return None

            frame, obj_names = _bake_queue.popleft()
            _bake_queue_item(scene, props, frame, obj_names)
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
    if props.use_frustum_cull and props.mode != 'ACTIVE':
        _region, rv3d = _get_active_3d_view()
        if rv3d:
            try:
                frustum_planes = _extract_frustum_planes(rv3d.perspective_matrix)
            except Exception:
                pass

    def _collect_ghost_parts(frames, cache, color_rgb, offset):
        """Collect (pos, vertex_color, offset_idx) tuples for a list of ghost frames."""
        parts = []
        n = len(frames)
        for i, frame in enumerate(frames):
            geo = cache.get(frame)
            if geo is None:
                continue
            pos, idx = geo
            n_verts = len(pos)
            if props.use_fade:
                t = (i + 1) / (n + 1)
                alpha = props.opacity * ((1.0 - t) ** props.fade_falloff)
            else:
                alpha = props.opacity
            vc = np.empty((n_verts, 4), dtype=np.float32)
            vc[:, :3] = color_rgb
            vc[:, 3] = alpha
            parts.append((pos, vc, idx + offset))
            offset += n_verts
        return parts, offset

    def _finalize_batch(parts):
        """Concatenate parts and create a GPU batch, or return None."""
        if not parts:
            return None
        m_pos = np.concatenate([p[0] for p in parts])
        m_col = np.concatenate([p[1] for p in parts])
        m_idx = np.concatenate([p[2] for p in parts])
        return batch_for_shader(
            _get_shader(), prim_type,
            {"pos": m_pos, "color": m_col},
            indices=m_idx.tolist(),
        )

    color_before = np.array(props.color_before[:3], dtype=np.float32)
    color_after = np.array(props.color_after[:3], dtype=np.float32)

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

        parts, before_offset = _collect_ghost_parts(before_frames, cache, color_before, before_offset)
        before_parts.extend(parts)

        parts, after_offset = _collect_ghost_parts(after_frames, cache, color_after, after_offset)
        after_parts.extend(parts)

    _merged_before = _finalize_batch(before_parts)
    _merged_after = _finalize_batch(after_parts)


def _build_occluder(scene, props, targets, depsgraph=None):
    """Build a world-space triangle occluder batch from current-frame meshes (MESH mode only).

    Clears the occluder when not in MESH mode or when there are no targets.
    Pass a depsgraph to reuse it (safe inside handlers); otherwise the current one is fetched.
    """
    global _occluder_batch
    _occluder_batch = None
    if props.in_front != 'MESH' or not props.enabled or not targets:
        return
    if depsgraph is None:
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except (AttributeError, RuntimeError):
            return
    parts_pos: list[np.ndarray] = []
    parts_idx: list[np.ndarray] = []
    offset = 0
    for obj in targets:
        try:
            geo = _bake_mesh_snapshot(obj, depsgraph, False, 1.0)  # always triangles, full detail
        except Exception:
            continue
        if geo is None:
            continue
        pos, idx = geo
        parts_pos.append(pos)
        parts_idx.append(idx + offset)
        offset += len(pos)
    if not parts_pos:
        return
    _occluder_batch = batch_for_shader(
        gpu.shader.from_builtin('UNIFORM_COLOR'), 'TRIS',
        {"pos": np.concatenate(parts_pos)},
        indices=np.concatenate(parts_idx).tolist(),
    )


def _shading_enabled(props, shading_type: str) -> bool:
    """Whether onion skin should draw in the current viewport shading type (per-shading filter)."""
    if shading_type == 'WIREFRAME':
        return props.show_in_wireframe
    if shading_type == 'SOLID':
        return props.show_in_solid
    if shading_type == 'MATERIAL':
        return props.show_in_material
    if shading_type == 'RENDERED':
        return props.show_in_rendered
    return True


def _draw_mesh_occluder():
    """Stamp the current mesh at near-plane depth (no color) so ghosts are always hidden behind it.

    Replaces projection row 2 with -row 3 to force NDC z to ~-1 (near plane) — no custom shader needed.
    Unlike show_in_front, works in all shading modes (solid / material / rendered).
    """
    if _occluder_batch is None:
        return
    proj = gpu.matrix.get_projection_matrix()
    rows = [list(proj[r]) for r in range(4)]
    r3 = rows[3]
    e = 0.9999  # just inside the near plane (exactly -1 may get clipped)
    rows[2] = [-r3[0] * e, -r3[1] * e, -r3[2] * e, -r3[3] * e]
    proj_near = Matrix(rows)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.depth_test_set('ALWAYS')
    gpu.state.depth_mask_set(True)
    gpu.state.color_mask_set(False, False, False, False)
    gpu.matrix.push_projection()
    try:
        gpu.matrix.load_projection_matrix(proj_near)
        shader.bind()
        shader.uniform_float("color", (0.0, 0.0, 0.0, 0.0))
        _occluder_batch.draw(shader)
    finally:
        gpu.matrix.pop_projection()
        gpu.state.color_mask_set(True, True, True, True)
        gpu.state.depth_mask_set(False)


def draw_onion_skins():
    """Viewport draw callback — renders 2 merged batches (before + after)."""
    global _merged_dirty
    try:
        scene = bpy.context.scene
        props = scene.mesh_onion_skin
    except (AttributeError, RuntimeError):
        return
    if not props.enabled:
        return

    # Per-shading-type visibility filter — check the shading type of the viewport being drawn
    try:
        shading_type = bpy.context.space_data.shading.type
    except AttributeError:
        shading_type = 'SOLID'
    if not _shading_enabled(props, shading_type):
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
    try:
        if props.in_front == 'GHOST':
            gpu.state.depth_test_set('NONE')
        elif props.in_front == 'MESH':
            # Stamp current mesh at near-plane depth first so ghosts fall behind it
            _draw_mesh_occluder()
            gpu.state.depth_test_set('LESS_EQUAL')
        else:  # NONE
            gpu.state.depth_test_set('LESS_EQUAL')
        if props.use_flat:
            gpu.state.line_width_set(1.5)

        shader.bind()
        if _merged_before is not None:
            _merged_before.draw(shader)
        if _merged_after is not None:
            _merged_after.draw(shader)
    finally:
        # Always restore GPU state (prevents a black viewport if drawing throws)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)
        gpu.state.color_mask_set(True, True, True, True)
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
def _on_depsgraph_update(scene, depsgraph):
    """Refresh ghosts/occluder when a target is edited without a frame change (manual posing, keyframe re-timing).

    - Occluder (MESH): rebuilt immediately from the current pose (cheap, no frame sampling — safe mid-drag).
    - Ghost cache: an actual animation-data edit (a target Action updated) invalidates the cache; the heavy
      rebuild is debounced until edits settle (_edit_settle_tick) so it never runs a frame_set mid-drag.
    (Pure frame changes go through _on_frame_change and never touch the Action datablock, so they are ignored here.)
    """
    global _building_occluder, _edit_seq, _edit_ack, _edit_rebuild_armed
    if _is_baking or _building_occluder:
        return
    try:
        props = scene.mesh_onion_skin
    except AttributeError:
        return
    if not props.enabled:
        return
    # Bail early if nothing geometric/transform changed and no Action was edited (ignore selection/property edits)
    if not any(u.is_updated_geometry or u.is_updated_transform or isinstance(u.id, bpy.types.Action)
               for u in depsgraph.updates):
        return
    targets = _collect_target_meshes(scene=scene)
    if not targets:
        return
    # Names of target meshes + their armatures, and the actions driving them
    names = {o.name for o in targets}
    action_names: set[str] = set()
    for o in targets:
        ad = o.animation_data
        if ad and ad.action:
            action_names.add(ad.action.name)  # object's own action (e.g. Keymesh)
        arm = _find_armature(o)
        if arm:
            names.add(arm.name)
            act = _get_active_action(arm)
            if act:
                action_names.add(act.name)
    # Separate "current pose moved" (drives the occluder) from "animation data edited" (drives the ghost rebuild)
    geo_changed = False
    anim_edited = False
    for upd in depsgraph.updates:
        idb = upd.id
        if isinstance(idb, bpy.types.Object) and idb.name in names \
                and (upd.is_updated_geometry or upd.is_updated_transform):
            geo_changed = True
        elif isinstance(idb, bpy.types.Action) and idb.name in action_names:
            anim_edited = True
    if not geo_changed and not anim_edited:
        return
    # MESH occluder follows the live pose (no frame sampling — safe during an active drag)
    if geo_changed and props.in_front == 'MESH':
        _building_occluder = True
        try:
            _build_occluder(scene, props, targets, depsgraph=depsgraph)
        finally:
            _building_occluder = False
    # Animation data edited → ghost geometry is stale. Debounce the heavy rebuild until edits settle.
    if anim_edited:
        _edit_seq += 1
        if not _edit_rebuild_armed:
            _edit_rebuild_armed = True
            _edit_ack = _edit_seq
            bpy.app.timers.register(_edit_settle_tick, first_interval=_EDIT_SETTLE)
    _request_viewport_redraw()


def _edit_settle_tick() -> float | None:
    """Rebuild ghost geometry once animation edits have settled (debounce)."""
    global _edit_rebuild_armed, _edit_ack
    # More edits since the last tick → keep waiting (never frame_set during an active drag)
    if _edit_seq != _edit_ack:
        _edit_ack = _edit_seq
        return _EDIT_SETTLE
    if _is_baking:
        return _EDIT_SETTLE  # a bake is running — retry shortly
    _edit_rebuild_armed = False
    try:
        scene = bpy.context.scene
        props = scene.mesh_onion_skin
    except (AttributeError, RuntimeError):
        return None
    if not props.enabled:
        return None
    targets = _collect_target_meshes(scene=scene)
    rebuild_cache(scene, targets, force_clear=True)
    _request_viewport_redraw()
    return None


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


_ONION_FCURVE_PATHS = (
    'mesh_onion_skin.use_keyframes',
    'mesh_onion_skin.use_flat',
    'mesh_onion_skin.mode',
)


def _clear_onion_fcurves(context):
    """Remove onion skin fcurves that would overwrite property values during frame_set()."""
    scene = context.scene if context else bpy.context.scene
    for path in _ONION_FCURVE_PATHS:
        _clear_fcurve_if_present(scene, path)


def _update_cache(self, context):
    """Incremental rebuild — reuses overlapping cached frames."""
    _clear_onion_fcurves(context)
    _schedule_rebuild(context)
    _request_viewport_redraw()


def _update_cache_full(self, context):
    """Full rebuild — clears all cache (for format changes like wireframe toggle)."""
    _clear_onion_fcurves(context)
    _schedule_rebuild(context, force_clear=True)
    _request_viewport_redraw()


def _update_mode(self, context):
    """On mode switch — full clear and rebuild (target set changes completely)."""
    _schedule_rebuild(context, force_clear=True)
    _request_viewport_redraw()


def _update_enabled(self, context):
    """On enable toggle — works for both the header checkbox and operator button."""
    if self.enabled:
        _schedule_rebuild(context, force_clear=True)
    else:
        clear_cache()
    _request_viewport_redraw()


def _update_display(self, context):
    """For settings that only need a redraw (color, opacity, fade change)."""
    global _merged_dirty
    _merged_dirty = True  # Colors/opacity baked into per-vertex data
    _request_viewport_redraw()


def _update_in_front(self, context):
    """On in-front mode change — rebuild/clear the MESH occluder (no show_in_front used).

    MESH is handled via a GPU occluder (near-plane depth), so it works in all shading modes.
    """
    _schedule_rebuild(context)
    _request_viewport_redraw()


def _update_redraw(self, context):
    """For display filters etc. that only need a redraw (no batch rebuild)."""
    _request_viewport_redraw()


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
    show_in_wireframe: BoolProperty(
        name="Wireframe View", default=True,
        description="Show onion skin in wireframe-shaded viewports",
        update=_update_redraw,
    )
    show_in_solid: BoolProperty(
        name="Solid View", default=True,
        description="Show onion skin in solid-shaded viewports",
        update=_update_redraw,
    )
    show_in_material: BoolProperty(
        name="Material Preview View", default=True,
        description="Show onion skin in material-preview viewports",
        update=_update_redraw,
    )
    show_in_rendered: BoolProperty(
        name="Rendered View", default=True,
        description="Show onion skin in rendered(-preview) viewports",
        update=_update_redraw,
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
        name="Off-Screen Skip", default=True,
        description="Skip drawing ghosts for objects outside the camera view",
        update=_update_display,
    )
    ghost_detail: FloatProperty(
        name="Ghost Detail", default=1.0, min=0.05, max=1.0,
        subtype='FACTOR',
        description="Reduce ghost triangle count for better performance (lower = fewer triangles)",
        update=_update_cache_full,
    )
    skip_same_pose: BoolProperty(
        name="Skip Same Pose", default=True,
        description="Hide ghosts that are identical to the current pose",
        update=_update_cache_full,
    )
    sync_bake: BoolProperty(
        name="Sync Bake (Live Follow)", default=False,
        description="Bake synchronously so ghosts follow during scrub and playback "
                    "(heavier per frame; playback/scrub may be less smooth)",
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
        _request_viewport_redraw()
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

    def draw(self, context):
        layout = self.layout
        props = context.scene.mesh_onion_skin

        # Enable/Disable button — always active regardless of enabled state
        header = layout.column()
        header.active = True
        row = header.row(align=True)
        toggle_text = "Disable" if props.enabled else "Enable"
        toggle_icon = 'PAUSE' if props.enabled else 'PLAY'
        row.operator("mesh.onion_skin_toggle", text=toggle_text,
                     icon=toggle_icon, depress=props.enabled)
        row.operator("mesh.onion_skin_update", text="", icon='FILE_REFRESH')

        # Grey out the rest when disabled
        col = layout.column()
        col.active = props.enabled
        layout = col

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

        box.prop(props, "use_keyframes")
        sub = box.row()
        sub.active = not props.use_keyframes
        sub.prop(props, "frame_step")
        if props.use_keyframes and props.mode == 'ACTIVE':
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
        box.prop(props, "in_front")
        box.prop(props, "use_flat")

        # Per-shading visibility filter — show ghosts only in the checked shading modes
        box.label(text="Show In")
        row = box.row(align=True)
        row.prop(props, "show_in_wireframe", text="", icon='SHADING_WIRE', toggle=True)
        row.prop(props, "show_in_solid", text="", icon='SHADING_SOLID', toggle=True)
        row.prop(props, "show_in_material", text="", icon='SHADING_TEXTURE', toggle=True)
        row.prop(props, "show_in_rendered", text="", icon='SHADING_RENDERED', toggle=True)

        # Color settings
        box = layout.box()
        box.label(text="Colors")
        row = box.row(align=True)
        row.prop(props, "color_before", text="")
        row.prop(props, "color_after", text="")

        # Performance settings
        box = layout.box()
        box.label(text="Performance")
        box.prop(props, "skip_same_pose")
        box.prop(props, "sync_bake")
        if props.mode != 'ACTIVE':
            box.prop(props, "ghost_detail", slider=True)
            box.prop(props, "bake_batch_size")
            box.prop(props, "use_frustum_cull")

        # Bake progress indicator
        if _bake_timer_running:
            box = layout.box()
            box.label(
                text=f"Baking... {_bake_progress:.0%}",
                icon='SORTTIME',
            )



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
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.load_post.append(_on_load_post)
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        draw_onion_skins, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _draw_handle, _rebuild_scheduled, _bake_timer_running, _bake_generation, _edit_rebuild_armed
    # Cancel progressive bake timer — increment generation so any pending timer self-aborts
    _bake_generation += 1
    _bake_queue.clear()
    _bake_timer_running = False
    # Cancel the edit-settle debounce timer
    _edit_rebuild_armed = False
    try:
        bpy.app.timers.unregister(_edit_settle_tick)
    except (ValueError, RuntimeError):
        pass
    if _rebuild_scheduled:
        try:
            bpy.app.timers.unregister(_do_rebuild)
        except (ValueError, RuntimeError):
            pass
        _rebuild_scheduled = False
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    clear_cache()
    bpy.app.handlers.load_post.remove(_on_load_post)
    bpy.app.handlers.frame_change_post.remove(_on_frame_change)
    try:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    except ValueError:
        pass
    del bpy.types.Scene.mesh_onion_skin
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
