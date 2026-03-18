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
# 전역 변수
# ---------------------------------------------------------------------------

# {오브젝트명: {프레임번호: GPUBatch}}
_onion_cache: dict[str, dict[int, gpu.types.GPUBatch]] = {}
_draw_handle = None
_is_baking = False
_rebuild_scheduled = False
_pending_rebuild = None  # (scene,)
# mesh_in_front 적용 전 원래 show_in_front 값 보존 {오브젝트명: bool}
_original_mesh_show_in_front: dict[str, bool] = {}

_PERF_WARN_THRESHOLD = 10


def _get_shader():
    return gpu.shader.from_builtin('UNIFORM_COLOR')


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
    """오브젝트에 애니메이션 데이터가 있는지 확인."""
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
    """GPU 배치 캐시 제거."""
    if obj_name:
        _onion_cache.pop(obj_name, None)
    else:
        _onion_cache.clear()


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


def _collect_keyframes_from_action(action) -> set[int]:
    """액션에서 키프레임 프레임 번호 수집 (Blender 5.0 Layered Action + 레거시 호환)."""
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
    # 레거시 폴백: action.fcurves 직접 접근
    if not kf_set:
        try:
            for fc in action.fcurves:
                for kp in fc.keyframe_points:
                    kf_set.add(round(kp.co[0]))
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
    kf_set = _collect_keyframes_from_action(action)
    if kf_set:
        return f"{arm.name} > {action.name}: {len(kf_set)}개", sorted(kf_set)
    return f"{arm.name} > {action.name}: 키프레임 0개", []


def _get_target_frames(scene, props, obj) -> list[int]:
    """고스트를 표시할 프레임 번호 목록 반환."""
    current = scene.frame_current
    frames: list[int] = []

    # 키프레임 모드는 Active 모드에서만 동작
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
# 베이킹
# ---------------------------------------------------------------------------

def _bake_mesh_snapshot(obj, depsgraph, use_flat: bool):
    """depsgraph가 이미 설정된 상태에서 메시 스냅샷을 GPU 배치로 반환."""
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
    """대상 오브젝트들의 어니언 스킨 캐시를 증분 빌드."""
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

    # 유효하지 않은 캐시 정리
    valid_names = {obj.name for obj in targets}
    stale = [k for k in _onion_cache if k not in valid_names]
    for k in stale:
        _onion_cache.pop(k, None)

    # 오브젝트별 타겟 프레임 수집 + 프레임-우선 베이킹 맵 구성
    obj_target_frames: dict[str, set[int]] = {}
    frames_to_objects: dict[int, list] = {}
    for obj in targets:
        frame_list = _get_target_frames(scene, props, obj)
        if not frame_list:
            clear_cache(obj.name)
            continue
        target_set = set(frame_list)
        obj_target_frames[obj.name] = target_set

        # 기존 캐시와 비교하여 베이킹 필요한 프레임만 수집
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

    # 프레임-우선 루프: frame_set 호출 최소화
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
# GPU 드로우
# ---------------------------------------------------------------------------

def draw_onion_skins():
    """뷰포트 드로우 콜백 – 캐시된 고스트 메시 렌더링."""
    scene = bpy.context.scene
    props = scene.mesh_onion_skin
    if not props.enabled:
        return

    # 캐시에 있는 오브젝트만 드로우 (rebuild_cache가 이미 수집/베이킹)
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
# 핸들러
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
    # 캐시 갱신 후 뷰포트 리드로우 요청
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
# Timer 기반 rebuild
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
    # 유효한 context가 있을 때 타겟을 미리 캡처
    targets = _collect_target_meshes(scene=scene, context=context)
    _pending_rebuild = (scene, targets)
    if not _rebuild_scheduled:
        _rebuild_scheduled = True
        bpy.app.timers.register(_do_rebuild, first_interval=0.0)


# ---------------------------------------------------------------------------
# 프로퍼티 업데이트 콜백
# ---------------------------------------------------------------------------

def _tag_redraw(context):
    if context and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


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


def _update_cache(self, context):
    # frame_set() 중 키프레임 데이터가 값을 덮어쓰는 것 방지: 관련 fcurve 제거
    scene = context.scene if context else bpy.context.scene
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.use_keyframes')
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.use_flat')
    _clear_fcurve_if_present(scene, 'mesh_onion_skin.mode')
    _schedule_rebuild(context)
    _tag_redraw(context)


def _update_mode(self, context):
    """모드 전환 시 – 캐시 클리어 + 리빌드."""
    _schedule_rebuild(context)
    _tag_redraw(context)


def _update_enabled(self, context):
    """활성화 토글 시 – 헤더 체크박스와 오퍼레이터 버튼 모두 동작."""
    if self.enabled:
        _schedule_rebuild(context)
    else:
    
        clear_cache()
    _tag_redraw(context)


def _update_display(self, context):
    """드로우만 갱신하면 되는 설정 변경 시."""
    _tag_redraw(context)


def _update_in_front(self, context):
    """앞에 표시 모드 변경 시 – 메쉬 show_in_front 적용/복원."""
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
        name="최대 오브젝트", default=10, min=1, max=50,
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
        update=_update_cache,
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
    use_flat: BoolProperty(
        name="와이어프레임", default=False,
        description="와이어프레임으로 고스트 표시",
        update=_update_cache,
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
        clear_cache()
        targets = _collect_target_meshes(context=context)
        rebuild_cache(context.scene, targets)
        _tag_redraw(context)
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

    def draw_header(self, context):
        props = context.scene.mesh_onion_skin
        self.layout.prop(props, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        props = context.scene.mesh_onion_skin
        layout.active = props.enabled

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

        if props.mode == 'ACTIVE':
            # 키프레임: Active 모드에서만 표시
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

        # 색상 설정
        box = layout.box()
        box.label(text="색상")
        row = box.row(align=True)
        row.prop(props, "color_before", text="")
        row.prop(props, "color_after", text="")

        # 액션 버튼
        row = layout.row(align=True)
        toggle_text = "비활성화" if props.enabled else "활성화"
        toggle_icon = 'PAUSE' if props.enabled else 'PLAY'
        row.operator("mesh.onion_skin_toggle", text=toggle_text,
                     icon=toggle_icon, depress=props.enabled)
        row.operator("mesh.onion_skin_update", text="", icon='FILE_REFRESH')


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
    # mesh_in_front로 변경된 show_in_front 복원
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
