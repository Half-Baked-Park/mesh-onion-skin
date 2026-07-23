bl_info = {
    "name": "Mesh Onion Skin",
    "author": "HB PARK",
    "version": (2, 3, 2),
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
# 전역 변수
# ---------------------------------------------------------------------------

# {오브젝트명: {프레임번호: (positions, indices)}}
_onion_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
# {아마추어명: (액션명, 정렬된 키프레임 리스트)} — 매 프레임 키프레임 재순회 방지
_keyframe_cache: dict[str, tuple[str, list[int]]] = {}
_draw_handle = None
_is_baking = False
_rebuild_scheduled = False
_pending_rebuild = None  # (scene,)

# --- 점진적 베이킹 상태 ---
_bake_queue: deque[tuple[int, list]] = deque()  # (frame, [obj_names]) 우선순위 큐
_bake_generation: int = 0  # 세대 카운터 — 스크러빙 시 stale 작업 취소
_bake_timer_running: bool = False
_bake_progress: float = 0.0  # 0.0~1.0 베이킹 진행률
_bake_total_frames: int = 0  # 현재 베이킹 작업의 총 프레임 수

# --- 머지드 배치 상태 (Phase 2) ---
_merged_before: gpu.types.GPUBatch | None = None
_merged_after: gpu.types.GPUBatch | None = None
_merged_dirty: bool = True

# --- 메쉬 앞(MESH) 오클루더: 현재 포즈를 근평면 깊이로 찍어 고스트를 가림 ---
# show_in_front과 달리 솔리드/머티리얼/렌더 모든 셰이딩 모드에서 동작
_occluder_batch: gpu.types.GPUBatch | None = None
_building_occluder: bool = False  # depsgraph 핸들러 재진입 방지

# --- 편집 감지 후 고스트 재빌드 디바운스 (키프레임 이동/포즈 편집) ---
_EDIT_SETTLE: float = 0.2  # 편집이 멎고 이 시간(초)이 지나면 재빌드 (드래그 중 frame_set 방지)
_edit_seq: int = 0         # 편집 감지마다 증가
_edit_ack: int = 0         # settle 타이머가 마지막으로 확인한 값
_edit_rebuild_armed: bool = False

# --- 디바운스 재빌드를 위한 '실제 편집' 감지 ---
# depsgraph는 키프레임이 실제로 안 바뀌어도 타겟 Action을 "업데이트됨"으로 표시함: 언두/리두의 데이터블록
# 복원, 포즈 모드 언두, 기타 헛된 재평가 모두 이 플래그를 올림. 이 오탐에 frame_set 재빌드를 무장하면
# fcurve가 재평가되어 키프레임 안 찍은 포즈 작업이 키프레임 값으로 덮어써짐(언두-포즈붕괴 버그). 그래서
# Action의 키프레임 *내용*이 실제로 바뀌었을 때만 재빌드를 무장 — Action별 시그니처로 추적.
_last_action_sig: dict[str, tuple] = {}

# 현재 베이크가 자신의 frame_set() 샘플링으로부터 보호해야 할 대상(포즈를 지킬 오브젝트들, 아니면 None).
# 각 베이크 블록이 시작 시 이들의 포즈를 *새로* 캡처하고 끝에 복원 — 전부 한 동기 블록 안이라 유저의
# 라이브 포징 위로 옛(stale) 포즈를 절대 안 씀. 프레임 변경 리빌드에선 None: 프레임이 바뀌면 포즈가 이미
# 재평가됐으니 지킬 언키 포즈가 없고, 스킵하면 복원의 matrix_basis 쓰기가 유발하는 프레임당 아마추어
# 재평가도 없앰.
_bake_capture_targets: list | None = None

# --- 동일 포즈 감지 (현재 프레임 스냅샷) ---
_current_frame_snapshots: dict[str, np.ndarray] = {}


def _get_shader():
    return gpu.shader.from_builtin('SMOOTH_COLOR')


# ---------------------------------------------------------------------------
# 프러스텀 컬링 (Phase 3)
# ---------------------------------------------------------------------------

def _get_active_3d_view():
    """활성 3D 뷰포트의 (region, region_3d) 반환. 없으면 (None, None)."""
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
    """VP 행렬에서 6개 프러스텀 평면 추출 (Gribb-Hartmann). (6, 4) 배열 반환."""
    m = np.array(perspective_matrix, dtype=np.float32)
    planes = np.empty((6, 4), dtype=np.float32)
    planes[0] = m[3] + m[0]   # Left
    planes[1] = m[3] - m[0]   # Right
    planes[2] = m[3] + m[1]   # Bottom
    planes[3] = m[3] - m[1]   # Top
    planes[4] = m[3] + m[2]   # Near
    planes[5] = m[3] - m[2]   # Far
    # 정규화
    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    planes /= norms
    return planes


def _is_in_frustum(obj, frustum_planes: np.ndarray) -> bool:
    """오브젝트의 바운딩 박스가 뷰 프러스텀과 교차하는지 테스트."""
    bb = obj.bound_box  # 로컬 좌표 8개 꼭짓점
    mat = np.array(obj.matrix_world, dtype=np.float32)
    corners_h = np.empty((8, 4), dtype=np.float32)
    corners_h[:, :3] = np.array(bb, dtype=np.float32)
    corners_h[:, 3] = 1.0
    world = (corners_h @ mat.T)[:, :3]  # (8, 3)
    # 8개 꼭짓점 모두 하나의 평면 바깥이면 → 프러스텀 밖
    for i in range(6):
        dots = world @ frustum_planes[i, :3] + frustum_planes[i, 3]
        if np.all(dots < 0):
            return False
    return True


# ---------------------------------------------------------------------------
# 타겟 수집
# ---------------------------------------------------------------------------

def _get_target_mesh(context=None):
    """활성 오브젝트 또는 포즈모드 아마추어의 메시 자식 반환."""
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
    """오브젝트에 애니메이션 소스가 있는지 확인 (액션, 드라이버, NLA, 제약조건, 애니메이션된 부모 포함)."""
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
    # 애니메이션된 부모의 자식 (예: 애니메이션된 Empty/Armature에 페어런트)
    if obj.parent and obj.parent.type != 'ARMATURE':
        return True
    return False


def _collect_target_meshes(scene=None, context=None) -> list:
    """모드에 따라 대상 메시 오브젝트 리스트 반환."""
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

    # SCENE / COLLECTION 모드
    if props.mode == 'COLLECTION':
        col = props.target_collection
        if col is None:
            return []
        source = col.all_objects
    else:  # SCENE
        source = scene.collection.all_objects

    candidates = [o for o in source if o.type == 'MESH']

    # 필터: 보이는 것 + 애니메이션 있는 것만
    candidates = [o for o in candidates if o.visible_get()]
    candidates = [o for o in candidates if _has_animation(o)]

    # 최대 개수 제한
    max_obj = props.max_objects
    if len(candidates) > max_obj:
        candidates = candidates[:max_obj]

    return candidates



# ---------------------------------------------------------------------------
# 캐시 관리
# ---------------------------------------------------------------------------

def clear_cache(obj_name: str | None = None):
    """지오메트리 캐시 제거 및 머지드 배치 무효화."""
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
    """오브젝트에 연결된 아마추어 반환 (부모 → 모디파이어 순)."""
    if obj.parent and obj.parent.type == 'ARMATURE':
        return obj.parent
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object:
            return mod.object
    return None


def _get_active_action(arm):
    """아마추어의 현재 활성 액션 반환 (직접 → NLA tweak → NLA 첫 스트립 순)."""
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
    """fcurve 하나의 키프레임 프레임 번호를 kf_set에 일괄 적재 (C-속도 foreach_get)."""
    n = len(fc.keyframe_points)
    if n == 0:
        return
    co = np.empty(n * 2, dtype=np.float32)
    fc.keyframe_points.foreach_get("co", co)
    # co = [frame0, value0, frame1, value1, ...] → 프레임 성분만 추출, 정수 반올림
    kf_set.update(np.rint(co[0::2]).astype(np.int64).tolist())


def _collect_keyframes_from_action(action) -> set[int]:
    """액션에서 키프레임 프레임 번호 수집 (Blender 5.0 Layered Action + 레거시 호환)."""
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
    # 레거시 폴백: action.fcurves 직접 접근
    if not kf_set:
        try:
            for fc in action.fcurves:
                _fcurve_key_frames(fc, kf_set)
        except (AttributeError, RuntimeError):
            pass
    return kf_set


def _get_armature_keyframes(obj) -> tuple[str, list[int]]:
    """아마추어의 현재 액션에서 키프레임 수집. (상태 문자열, 프레임 리스트) 반환."""
    arm = _find_armature(obj)
    if arm is None:
        return "아마추어 없음", []
    action = _get_active_action(arm)
    if action is None:
        return f"{arm.name}: 활성 액션 없음", []
    # 아마추어별 캐시 — 액션이 바뀌거나 캐시가 클리어될 때만 재수집.
    # (같은 액션에서 키프레임을 수정한 뒤엔 Update 버튼으로 갱신.)
    cached = _keyframe_cache.get(arm.name)
    if cached is not None and cached[0] == action.name:
        frames = cached[1]
    else:
        frames = sorted(_collect_keyframes_from_action(action))
        _keyframe_cache[arm.name] = (action.name, frames)
    if frames:
        return f"{arm.name} > {action.name}: {len(frames)}개", frames
    return f"{arm.name} > {action.name}: 키프레임 0개", []


def _get_target_frames(scene, props, obj) -> list[int]:
    """고스트를 표시할 프레임 번호 목록 반환."""
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
# 베이킹
# ---------------------------------------------------------------------------

def _bake_mesh_snapshot(obj, depsgraph, use_flat: bool, ghost_detail: float = 1.0):
    """메시 스냅샷의 (positions, indices) numpy 배열 반환."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    if mesh is None or len(mesh.vertices) == 0:
        eval_obj.to_mesh_clear()
        return None

    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)

    # 월드 좌표 변환 — pre-allocated homogeneous 좌표 (hstack 크래시 방지)
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

    # Ghost Detail — 균일 샘플링으로 삼각형/엣지 수 축소
    if ghost_detail < 1.0 and len(idx) > 1:
        keep = max(1, int(len(idx) * ghost_detail))
        step = max(1, len(idx) // keep)
        idx = idx[::step][:keep]

    return (co, idx)


def _build_prioritized_queue(current_frame: int, frames_to_objects: dict[int, list]) -> list[tuple[int, list]]:
    """현재 프레임에 가까운 순으로 정렬. 가까운 프레임부터 베이킹."""
    items = list(frames_to_objects.items())
    items.sort(key=lambda pair: abs(pair[0] - current_frame))
    return items


def _bake_queue_item(scene, props, frame, obj_names) -> None:
    """(frame, obj_names) 큐 항목 하나를 캐시에 베이킹. 동기/점진 경로 공유.

    _is_baking 설정과 현재 프레임 복원은 호출자 책임.
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
        # 현재 프레임과 동일한 포즈면 스킵
        if props.skip_same_pose:
            cur_snap = _current_frame_snapshots.get(obj_name)
            if cur_snap is not None and cur_snap.shape == geo[0].shape:
                if np.allclose(cur_snap, geo[0], atol=1e-4):
                    continue
        _onion_cache.setdefault(obj.name, {})[frame] = geo
        _merged_dirty = True


def _capture_pose_state(targets):
    """베이크의 scene.frame_set()이 재평가로 날려버릴 라이브 로컬 트랜스폼을 스냅샷.

    베이크는 scene.frame_set()으로 고스트 프레임을 샘플링하는데, 이때 모든 fcurve가 재평가되어 키프레임
    안 찍은 포즈(포징했지만 미키프레임)가 키프레임 값으로 조용히 덮어써짐. 대상 아마추어 포즈본/오브젝트의
    matrix_basis를 베이크 전에 캡처했다가 후에 복원해 그 작업을 보존. 복원 토큰(list) 또는 None 반환.
    """
    state = []
    seen: set[str] = set()

    def _grab(obj):
        if obj is None or obj.name in seen:
            return
        seen.add(obj.name)
        try:
            bones = ([(pb.name, pb.matrix_basis.copy()) for pb in obj.pose.bones]
                     if obj.type == 'ARMATURE' and obj.pose else None)
            state.append((obj, bones, obj.matrix_basis.copy()))
        except (AttributeError, ReferenceError):
            pass

    for o in targets:
        _grab(o)
        _grab(_find_armature(o))
    return state or None


def _restore_pose_state(state):
    """_capture_pose_state가 캡처한 트랜스폼을 되쓰기. _is_baking 구간 안에서 호출."""
    if not state:
        return
    for obj, bones, obj_basis in state:
        try:
            obj.matrix_basis = obj_basis
            if bones:
                pbs = obj.pose.bones
                for name, mb in bones:
                    pb = pbs.get(name)
                    if pb is not None:
                        pb.matrix_basis = mb
        except (ReferenceError, RuntimeError, AttributeError):
            pass


def _bake_all_sync(scene, props) -> None:
    """큐 전체를 즉시(블로킹) 베이킹. sync_bake(실시간 추종) 켜졌을 때 사용."""
    global _is_baking
    current = scene.frame_current
    _is_baking = True
    pose = _capture_pose_state(_bake_capture_targets) if _bake_capture_targets else None
    try:
        while _bake_queue:
            frame, obj_names = _bake_queue.popleft()
            _bake_queue_item(scene, props, frame, obj_names)
    finally:
        # 현재 프레임 + 이 동기 블록 맨 위에서 캡처한 포즈 복원
        try:
            scene.frame_set(current)
        except Exception:
            pass
        _restore_pose_state(pose)
        _is_baking = False


def rebuild_cache(scene, targets=None, force_clear: bool = False, capture_pose: bool = True):
    """델타 계산 후 점진적 베이킹 큐에 등록. 논블로킹.

    capture_pose: 베이크의 frame_set 전후로 라이브 포즈를 스냅샷/복원해 언키 포즈가 안 덮이게 함.
    프레임 변경 리빌드에선 False로 — 거기선 언키 포즈가 있을 수 없고, 프레임당 비용도 아낌.
    """
    global _bake_generation, _bake_timer_running, _bake_progress, _bake_total_frames, _merged_dirty
    global _bake_capture_targets
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

    # 포맷 변경 시 전체 클리어 (예: 와이어프레임 토글)
    if force_clear:
        clear_cache()

    # 유효하지 않은 캐시 정리
    valid_names = {obj.name for obj in targets}
    stale = [k for k in _onion_cache if k not in valid_names]
    for k in stale:
        _onion_cache.pop(k, None)

    # 현재 포즈 오클루더 재빌드 (MESH 모드 전용, 근평면 깊이용)
    _build_occluder(scene, props, targets)

    # 편집 감지 기준선 콜드스타트 프라이밍(1회) — (재)빌드 후 첫 실제 편집을 놓치지 않게.
    # undo는 이 값을 동일하게 두므로 무연산 유지; 이후엔 _on_depsgraph_update가 유지 관리.
    for obj in targets:
        ad = obj.animation_data
        if ad and ad.action and ad.action.name not in _last_action_sig:
            _last_action_sig[ad.action.name] = _action_signature(ad.action)
        arm = _find_armature(obj)
        if arm:
            act = _get_active_action(arm)
            if act and act.name not in _last_action_sig:
                _last_action_sig[act.name] = _action_signature(act)

    # 오브젝트별 타겟 프레임 수집 + 델타만 베이킹 맵 구성
    frames_to_objects: dict[int, list] = {}
    for obj in targets:
        frame_list = _get_target_frames(scene, props, obj)
        if not frame_list:
            clear_cache(obj.name)
            continue

        # 기존 유효 캐시 보존, 불필요 프레임 제거
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

    # 진행 중인 베이킹 취소
    _bake_generation += 1

    # 베이크가 자신의 frame_set으로부터 보호할 대상 기록 (실제 캡처는 아래 각 베이크 블록에서 새로)
    _bake_capture_targets = targets if capture_pose else None

    # 동일 포즈 감지를 위해 현재 프레임 스냅샷 캡처 (리빌드당 1회)
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

    # 우선순위 큐 구성 — 가까운 프레임부터
    _bake_queue.clear()
    _bake_queue.extend(_build_prioritized_queue(scene.frame_current, frames_to_objects))
    _bake_total_frames = len(_bake_queue)
    _bake_progress = 0.0

    # 동기(실시간 추종) 베이크 — 지금 전부 베이킹해 스크럽/재생 중에도 고스트가 따라옴
    if props.sync_bake:
        _bake_all_sync(scene, props)
        _bake_timer_running = False
        _bake_progress = 1.0
        _merged_dirty = True
        return

    # 점진적 베이킹 타이머 시작
    # 항상 새 타이머 등록 — 기존 타이머는 세대 불일치로 자동 중단
    _bake_timer_running = True
    gen = _bake_generation
    bpy.app.timers.register(
        partial(_progressive_bake_tick, gen),
        first_interval=0.0,
    )


def _progressive_bake_tick(generation: int) -> float | None:
    """타이머 콜백 — 틱당 N 프레임 베이킹, Blender에 제어 반환."""
    global _is_baking, _bake_timer_running, _bake_progress

    # 세대 불일치 — 중단
    if generation != _bake_generation:
        _bake_timer_running = False
        return None

    # 큐 비어있음 — 완료
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

    # 배치 크기 결정 (틱당 프레임 수)
    batch_size = max(1, props.bake_batch_size)
    current = scene.frame_current
    _is_baking = True
    pose = _capture_pose_state(_bake_capture_targets) if _bake_capture_targets else None

    try:
        frames_done = 0
        while _bake_queue and frames_done < batch_size:
            # 루프 내 세대 재확인
            if generation != _bake_generation:
                _bake_timer_running = False
                return None

            frame, obj_names = _bake_queue.popleft()
            _bake_queue_item(scene, props, frame, obj_names)
            frames_done += 1
    finally:
        # 현재 프레임 + 이 틱 맨 위에서 캡처한 포즈 복원 (fresh — stale 아님)
        try:
            scene.frame_set(current)
        except Exception:
            pass
        _restore_pose_state(pose)
        _is_baking = False

    # 진행률 갱신
    if _bake_total_frames > 0:
        done = _bake_total_frames - len(_bake_queue)
        _bake_progress = done / _bake_total_frames

    # 새로 베이킹된 고스트 표시를 위해 뷰포트 리드로우 요청
    _request_viewport_redraw()

    if _bake_queue:
        return 0.0  # 즉시 다음 틱 예약
    else:
        _bake_timer_running = False
        _bake_progress = 1.0
        return None


def _request_viewport_redraw():
    """모든 3D 뷰포트에 리드로우 요청."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GPU 드로우 — 머지드 배치 시스템
# ---------------------------------------------------------------------------

def _build_merged_batches():
    """캐시된 모든 고스트 지오메트리를 2개의 mega-batch로 합침 (before + after)."""
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

    # 그룹별 지오메트리 수집
    before_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    after_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    before_offset = 0
    after_offset = 0

    # 프러스텀 컬링 — 뷰포트 밖 오브젝트 스킵
    frustum_planes = None
    if props.use_frustum_cull and props.mode != 'ACTIVE':
        _region, rv3d = _get_active_3d_view()
        if rv3d:
            try:
                frustum_planes = _extract_frustum_planes(rv3d.perspective_matrix)
            except Exception:
                pass

    def _collect_ghost_parts(frames, cache, color_rgb, offset):
        """고스트 프레임 목록에서 (pos, vertex_color, offset_idx) 튜플 수집."""
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
        """파트를 합치고 GPU 배치 생성. 없으면 None 반환."""
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

        # 프러스텀 컬링 — 카메라 밖 오브젝트 스킵
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
    """현재 프레임 메쉬들을 월드 트라이앵글 오클루더 배치로 빌드 (MESH 모드 전용).

    MESH 모드가 아니거나 타겟이 없으면 오클루더를 비운다.
    depsgraph를 넘기면 그것을 사용(핸들러 내 재진입 안전), 없으면 현재 depsgraph 조회.
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
            geo = _bake_mesh_snapshot(obj, depsgraph, False, 1.0)  # 항상 삼각형, 풀 디테일
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
    """현재 뷰포트 셰이딩 타입에서 어니언 스킨을 표시할지 (셰이더 종류별 필터)."""
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
    """현재 메쉬를 근평면 깊이로 찍어(색 안 씀) 고스트가 항상 메쉬 뒤로 가려지게 함.

    투영행렬 2행을 -3행으로 바꿔 NDC z를 ~-1(근평면)로 고정 → 커스텀 셰이더 불필요.
    show_in_front과 달리 솔리드/머티리얼/렌더 모든 셰이딩 모드에서 동작.
    """
    if _occluder_batch is None:
        return
    proj = gpu.matrix.get_projection_matrix()
    rows = [list(proj[r]) for r in range(4)]
    r3 = rows[3]
    e = 0.9999  # 근평면 바로 안쪽 (정확히 -1이면 클리핑될 수 있음)
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
    """뷰포트 드로우 콜백 — 머지드 배치 2개만 드로우 (before + after)."""
    global _merged_dirty
    try:
        scene = bpy.context.scene
        props = scene.mesh_onion_skin
    except (AttributeError, RuntimeError):
        return
    if not props.enabled:
        return

    # 셰이더 종류별 표시 필터 — 지금 그려지는 뷰포트의 셰이딩 타입 확인
    try:
        shading_type = bpy.context.space_data.shading.type
    except AttributeError:
        shading_type = 'SOLID'
    if not _shading_enabled(props, shading_type):
        return

    if not _onion_cache:
        return

    # 지오메트리나 디스플레이 설정 변경 시 머지드 배치 재빌드
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
            # 현재 메쉬를 근평면 깊이로 먼저 찍어 고스트를 그 뒤로 가림
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
        # GPU 상태 항상 복원 (예외 시 뷰포트가 검게 깨지는 것 방지)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)
        gpu.state.color_mask_set(True, True, True, True)
        if props.use_flat:
            gpu.state.line_width_set(1.0)


# ---------------------------------------------------------------------------
# 핸들러
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
    rebuild_cache(scene, targets, capture_pose=False)  # 프레임 변경 → 지킬 언키 포즈 없음
    _merged_dirty = True  # 알파 값이 현재 프레임에 따라 변경됨
    _request_viewport_redraw()


def _action_signature(action) -> tuple:
    """Action 키프레임 내용의 지문(키프레임 개수 + 순서무관 내용 해시).

    키프레임 추가/삭제/이동(리타이밍)·값 편집 시 바뀌고, depsgraph가 실제 편집 없이 Action을 그냥
    "업데이트됨"으로 다시 표시할 때(언두/리두, 포즈 모드 언두)는 동일하게 유지됨. 진짜 애니 편집과
    헛된 업데이트를 구분해, 파괴적 frame_set 재빌드를 전자에만 돌리기 위함. 채널별 해시에 채널 식별자를
    섞어, 합이 보존되는 편집(예: +1/−1 키 이동, 채널 간 값 맞바꿈)도 감지됨.
    """
    count = 0
    h = 0

    def _accum(fcurves) -> int:
        nonlocal count, h
        n = 0
        for fc in fcurves:
            n += 1
            kfs = fc.keyframe_points
            m = len(kfs)
            if not m:
                continue
            arr = np.empty(m * 2, dtype=np.float64)
            kfs.foreach_get('co', arr)
            count += m
            h ^= hash((fc.data_path, fc.array_index, arr.tobytes()))
        return n

    layered = 0
    try:
        for layer in action.layers:            # Blender 5.0+ 레이어드 액션
            for strip in layer.strips:
                for bag in strip.channelbags:
                    layered += _accum(bag.fcurves)
    except AttributeError:
        pass
    if layered == 0:                           # 레이어드 fcurve가 없을 때만 레거시 폴백
        try:
            _accum(action.fcurves)
        except (AttributeError, RuntimeError):
            pass
    return (count, h)


@persistent
def _on_depsgraph_update(scene, depsgraph):
    """프레임 이동 없이 타겟을 편집(수동 포징/키프레임 리타이밍)했을 때 고스트·오클루더를 갱신.

    - 오클루더(MESH): 현재 포즈로 즉시 재빌드 (프레임 샘플링 없어 드래그 중에도 안전).
    - 고스트 캐시: 실제 애니 편집(타겟 Action 업데이트)이면 캐시 무효화 후 재빌드하되,
      재빌드는 frame_set을 쓰므로 편집이 멎을 때까지 디바운스 (_edit_settle_tick).
    (순수 프레임 변경은 _on_frame_change가 처리하고 Action을 안 건드리므로 여기서 무시됨.)
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
    # 지오/트랜스폼 변경도 Action 편집도 없으면 조기 종료 (선택·프로퍼티 변경 등 무시)
    if not any(u.is_updated_geometry or u.is_updated_transform or isinstance(u.id, bpy.types.Action)
               for u in depsgraph.updates):
        return
    targets = _collect_target_meshes(scene=scene)
    if not targets:
        return
    # 타겟 메쉬/아마추어 이름 + 관련 액션 이름 수집
    names = {o.name for o in targets}
    action_names: set[str] = set()
    for o in targets:
        ad = o.animation_data
        if ad and ad.action:
            action_names.add(ad.action.name)  # 오브젝트 자체 액션 (예: Keymesh)
        arm = _find_armature(o)
        if arm:
            names.add(arm.name)
            act = _get_active_action(arm)
            if act:
                action_names.add(act.name)
    # 현재 포즈가 움직였나(오클루더용) vs 애니 데이터가 편집됐나(고스트 재빌드용) 구분
    geo_changed = False
    anim_edited = False
    for upd in depsgraph.updates:
        idb = upd.id
        if isinstance(idb, bpy.types.Object) and idb.name in names \
                and (upd.is_updated_geometry or upd.is_updated_transform):
            geo_changed = True
        elif isinstance(idb, bpy.types.Action) and idb.name in action_names:
            # "업데이트됨"만으론 언두/리두 오탐 — 실제 키프레임 변화가 있을 때만 인정.
            sig = _action_signature(idb)
            prev = _last_action_sig.get(idb.name)
            _last_action_sig[idb.name] = sig
            if prev is not None and prev != sig:
                anim_edited = True
    if not geo_changed and not anim_edited:
        return
    # MESH 오클루더는 현재 포즈를 따라감 (frame_set 없음 → 드래그 중에도 안전)
    if geo_changed and props.in_front == 'MESH':
        _building_occluder = True
        try:
            _build_occluder(scene, props, targets, depsgraph=depsgraph)
        finally:
            _building_occluder = False
    # 실제 키프레임 편집 → 고스트 지오메트리 stale. 편집이 멎을 때까지 디바운스 후 재빌드
    if anim_edited:
        _edit_seq += 1
        if not _edit_rebuild_armed:
            _edit_rebuild_armed = True
            _edit_ack = _edit_seq
            bpy.app.timers.register(_edit_settle_tick, first_interval=_EDIT_SETTLE)
    _request_viewport_redraw()


def _edit_settle_tick() -> float | None:
    """애니 편집이 멎은 뒤(디바운스) 고스트 지오메트리를 재빌드."""
    global _edit_rebuild_armed, _edit_ack
    # 지난 tick 이후 편집이 더 들어왔으면 계속 대기 (드래그 중 frame_set 금지)
    if _edit_seq != _edit_ack:
        _edit_ack = _edit_seq
        return _EDIT_SETTLE
    if _is_baking:
        return _EDIT_SETTLE  # 베이크 진행 중 — 잠시 후 재시도
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
    _last_action_sig.clear()  # 이전 파일의 stale 편집감지 기준값 폐기
    clear_cache()


# ---------------------------------------------------------------------------
# Timer 기반 rebuild
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
    """다음 타이머 틱에 리빌드 예약. 기본적으로 캐시를 클리어하지 않음."""
    global _rebuild_scheduled, _pending_rebuild
    try:
        scene = context.scene if context else bpy.context.scene
    except AttributeError:
        return
    # 유효한 context가 있을 때 타겟을 미리 캡처
    targets = _collect_target_meshes(scene=scene, context=context)
    _pending_rebuild = (scene, targets, force_clear)
    if not _rebuild_scheduled:
        _rebuild_scheduled = True
        bpy.app.timers.register(_do_rebuild, first_interval=0.0)


# ---------------------------------------------------------------------------
# 프로퍼티 업데이트 콜백
# ---------------------------------------------------------------------------

def _clear_fcurve_if_present(scene, data_path: str):
    """씬 액션에서 해당 경로의 fcurve를 제거한다. frame_set() 중 덮어쓰기 방지용."""
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
    # 레거시 폴백
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
    """frame_set() 중 프로퍼티 값 덮어쓰기 방지를 위해 어니언 스킨 fcurve 제거."""
    scene = context.scene if context else bpy.context.scene
    for path in _ONION_FCURVE_PATHS:
        _clear_fcurve_if_present(scene, path)


def _update_cache(self, context):
    """증분 리빌드 — 겹치는 캐시 프레임 재활용."""
    _clear_onion_fcurves(context)
    _schedule_rebuild(context)
    _request_viewport_redraw()


def _update_cache_full(self, context):
    """전체 리빌드 — 포맷 변경 시 전체 캐시 클리어 (와이어프레임 토글 등)."""
    _clear_onion_fcurves(context)
    _schedule_rebuild(context, force_clear=True)
    _request_viewport_redraw()


def _update_mode(self, context):
    """모드 전환 시 – 전체 클리어 + 리빌드 (타겟 세트 완전 변경)."""
    _schedule_rebuild(context, force_clear=True)
    _request_viewport_redraw()


def _update_enabled(self, context):
    """활성화 토글 시 – 헤더 체크박스와 오퍼레이터 버튼 모두 동작."""
    if self.enabled:
        _schedule_rebuild(context, force_clear=True)
    else:
        clear_cache()
    _request_viewport_redraw()


def _update_display(self, context):
    """드로우만 갱신하면 되는 설정 변경 시 (색상, 불투명도, 페이드)."""
    global _merged_dirty
    _merged_dirty = True  # 색상/불투명도가 per-vertex 데이터에 반영됨
    _request_viewport_redraw()


def _update_in_front(self, context):
    """앞에 표시 모드 변경 시 – MESH 오클루더 재빌드/해제 (show_in_front 미사용).

    MESH는 GPU 오클루더(근평면 깊이)로 처리하므로 모든 셰이딩 모드에서 동작.
    """
    _schedule_rebuild(context)
    _request_viewport_redraw()


def _update_redraw(self, context):
    """표시 필터 등 드로우만 갱신하면 되는 경우 (배치 재빌드 불필요)."""
    _request_viewport_redraw()


# ---------------------------------------------------------------------------
# 프로퍼티 그룹
# ---------------------------------------------------------------------------

class MeshOnionSkinProps(PropertyGroup):
    enabled: BoolProperty(
        name="활성화",
        description="어니언 스킨 표시",
        default=False,
        update=_update_enabled,
    )
    mode: EnumProperty(
        name="모드",
        items=[
            ('ACTIVE', "활성", "활성 오브젝트만 고스트 표시", 'OBJECT_DATA', 0),
            ('SCENE', "씬", "씬의 모든 메쉬에 고스트 표시", 'SCENE_DATA', 1),
            ('COLLECTION', "콜렉션", "콜렉션의 모든 메쉬에 고스트 표시", 'OUTLINER_COLLECTION', 2),
        ],
        default='ACTIVE',
        update=_update_mode,
    )
    target_collection: PointerProperty(
        type=bpy.types.Collection,
        name="콜렉션",
        description="어니언 스킨 대상 콜렉션",
        update=_update_cache,
    )
    max_objects: IntProperty(
        name="최대 오브젝트", default=10, min=1, max=500,
        description="씬/콜렉션 모드에서 처리할 최대 오브젝트 수",
        update=_update_cache,
    )
    count_before: IntProperty(
        name="이전", default=3, min=0, max=10,
        description="현재 프레임 이전 고스트 수",
        update=_update_cache,
    )
    count_after: IntProperty(
        name="이후", default=3, min=0, max=10,
        description="현재 프레임 이후 고스트 수",
        update=_update_cache,
    )
    frame_step: IntProperty(
        name="간격", default=1, min=1, max=10,
        description="프레임 간격",
        update=_update_cache,
    )
    use_keyframes: BoolProperty(
        name="키프레임만", default=False,
        description="아마추어 키프레임 위치에만 고스트 표시 (Active 모드 전용)",
        update=_update_cache_full,
    )
    color_before: FloatVectorProperty(
        name="이전 색상", subtype='COLOR_GAMMA',
        size=3, default=(0.2, 0.8, 0.2), min=0.0, max=1.0,
        description="이전 프레임 고스트 색상",
        update=_update_display,
    )
    color_after: FloatVectorProperty(
        name="이후 색상", subtype='COLOR_GAMMA',
        size=3, default=(0.2, 0.4, 0.9), min=0.0, max=1.0,
        description="이후 프레임 고스트 색상",
        update=_update_display,
    )
    opacity: FloatProperty(
        name="불투명도", default=0.5, min=0.0, max=1.0,
        subtype='FACTOR',
        description="고스트 불투명도",
        update=_update_display,
    )
    use_fade: BoolProperty(
        name="페이드", default=True,
        description="거리에 따라 불투명도 감소",
        update=_update_display,
    )
    fade_falloff: FloatProperty(
        name="페이드 강도", default=1.0, min=0.2, max=5.0,
        subtype='FACTOR',
        description="페이드 커브 강도 (높을수록 가까운 고스트와 먼 고스트의 차이가 커짐)",
        update=_update_display,
    )
    in_front: EnumProperty(
        name="앞에 표시",
        items=[
            ('NONE', "없음", "기본 깊이 테스트 사용"),
            ('GHOST', "고스트", "고스트를 항상 앞에 표시"),
            ('MESH', "메쉬", "메쉬 오브젝트를 항상 앞에 표시"),
        ],
        default='GHOST',
        update=_update_in_front,
    )
    show_in_wireframe: BoolProperty(
        name="와이어프레임 뷰", default=True,
        description="와이어프레임 셰이딩 뷰포트에서 어니언 스킨 표시",
        update=_update_redraw,
    )
    show_in_solid: BoolProperty(
        name="솔리드 뷰", default=True,
        description="솔리드 셰이딩 뷰포트에서 어니언 스킨 표시",
        update=_update_redraw,
    )
    show_in_material: BoolProperty(
        name="머티리얼 미리보기 뷰", default=True,
        description="머티리얼 미리보기 뷰포트에서 어니언 스킨 표시",
        update=_update_redraw,
    )
    show_in_rendered: BoolProperty(
        name="렌더 뷰", default=True,
        description="렌더(미리보기) 뷰포트에서 어니언 스킨 표시",
        update=_update_redraw,
    )
    use_flat: BoolProperty(
        name="와이어프레임", default=False,
        description="와이어프레임으로 고스트 표시",
        update=_update_cache_full,
    )
    bake_batch_size: IntProperty(
        name="베이크 배치", default=2, min=1, max=10,
        description="타이머 틱당 베이킹할 프레임 수 (높을수록 빠르지만 버벅임 증가)",
    )
    use_frustum_cull: BoolProperty(
        name="화면 밖 스킵", default=True,
        description="카메라 밖 오브젝트의 고스트 드로우 스킵",
        update=_update_display,
    )
    ghost_detail: FloatProperty(
        name="고스트 디테일", default=1.0, min=0.05, max=1.0,
        subtype='FACTOR',
        description="고스트 삼각형 수 축소로 성능 향상 (낮을수록 삼각형 적음)",
        update=_update_cache_full,
    )
    skip_same_pose: BoolProperty(
        name="동일 포즈 스킵", default=True,
        description="현재 포즈와 동일한 고스트 숨기기",
        update=_update_cache_full,
    )
    sync_bake: BoolProperty(
        name="동기 베이크 (실시간 추종)", default=False,
        description="동기로 베이킹해 스크럽·재생 중에도 고스트가 따라옴 "
                    "(프레임당 더 무거움; 재생·스크럽이 덜 부드러울 수 있음)",
    )


# ---------------------------------------------------------------------------
# 오퍼레이터
# ---------------------------------------------------------------------------

class MESH_OT_onion_skin_toggle(Operator):
    bl_idname = "mesh.onion_skin_toggle"
    bl_label = "어니언 스킨 토글"
    bl_description = "어니언 스킨 표시 토글"

    def execute(self, context):
        props = context.scene.mesh_onion_skin
        props.enabled = not props.enabled
        return {'FINISHED'}


class MESH_OT_onion_skin_update(Operator):
    bl_idname = "mesh.onion_skin_update"
    bl_label = "어니언 스킨 갱신"
    bl_description = "어니언 스킨 캐시 강제 갱신"

    def execute(self, context):
        targets = _collect_target_meshes(context=context)
        rebuild_cache(context.scene, targets, force_clear=True)
        _request_viewport_redraw()
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# 패널
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

        # 활성화/비활성화 버튼 — enabled 상태와 무관하게 항상 활성
        header = layout.column()
        header.active = True
        row = header.row(align=True)
        toggle_text = "비활성화" if props.enabled else "활성화"
        toggle_icon = 'PAUSE' if props.enabled else 'PLAY'
        row.operator("mesh.onion_skin_toggle", text=toggle_text,
                     icon=toggle_icon, depress=props.enabled)
        row.operator("mesh.onion_skin_update", text="", icon='FILE_REFRESH')

        # 비활성화 시 나머지 UI 회색 처리
        col = layout.column()
        col.active = props.enabled
        layout = col

        # 모드 선택
        layout.prop(props, "mode", text="")

        # 대상 설정 (Scene/Collection 모드에서만)
        if props.mode != 'ACTIVE':
            box = layout.box()
            box.label(text="대상")
            if props.mode == 'COLLECTION':
                box.prop(props, "target_collection", text="")
            box.prop(props, "max_objects")
            targets = _collect_target_meshes(context=context)
            count = len(targets)
            box.label(
                text=f"  {count}개 오브젝트",
                icon='MESH_DATA',
            )
            if count >= props.max_objects:
                box.label(
                    text="  최대 개수 도달",
                    icon='ERROR',
                )

        # 프레임 설정
        box = layout.box()
        box.label(text="프레임")
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

        # 표시 설정
        box = layout.box()
        box.label(text="표시")
        box.prop(props, "opacity", slider=True)
        box.prop(props, "use_fade")
        sub = box.row()
        sub.active = props.use_fade
        sub.prop(props, "fade_falloff", slider=True)
        box.prop(props, "in_front")
        box.prop(props, "use_flat")

        # 표시할 뷰포트 셰이딩 필터 — 체크한 셰이딩 모드에서만 고스트 표시
        box.label(text="표시 뷰")
        row = box.row(align=True)
        row.prop(props, "show_in_wireframe", text="", icon='SHADING_WIRE', toggle=True)
        row.prop(props, "show_in_solid", text="", icon='SHADING_SOLID', toggle=True)
        row.prop(props, "show_in_material", text="", icon='SHADING_TEXTURE', toggle=True)
        row.prop(props, "show_in_rendered", text="", icon='SHADING_RENDERED', toggle=True)

        # 색상 설정
        box = layout.box()
        box.label(text="색상")
        row = box.row(align=True)
        row.prop(props, "color_before", text="")
        row.prop(props, "color_after", text="")

        # 성능 설정
        box = layout.box()
        box.label(text="성능")
        box.prop(props, "skip_same_pose")
        box.prop(props, "sync_bake")
        if props.mode != 'ACTIVE':
            box.prop(props, "ghost_detail", slider=True)
            box.prop(props, "bake_batch_size")
            box.prop(props, "use_frustum_cull")

        # 베이킹 진행률 표시
        if _bake_timer_running:
            box = layout.box()
            box.label(
                text=f"베이킹 중... {_bake_progress:.0%}",
                icon='SORTTIME',
            )



# ---------------------------------------------------------------------------
# 등록
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
    # 점진적 베이킹 타이머 정리 — 세대 증가로 pending 타이머 즉시 중단
    _bake_generation += 1
    _bake_queue.clear()
    _bake_timer_running = False
    # 편집 디바운스 타이머 정리
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
    for _hlist, _h in ((bpy.app.handlers.load_post, _on_load_post),
                       (bpy.app.handlers.frame_change_post, _on_frame_change),
                       (bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update)):
        try:
            _hlist.remove(_h)
        except ValueError:
            pass
    del bpy.types.Scene.mesh_onion_skin
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
