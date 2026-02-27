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
# 전역 변수
# ---------------------------------------------------------------------------

# {오브젝트명: {프레임번호: GPUBatch}}
_onion_cache: dict[str, dict[int, gpu.types.GPUBatch]] = {}
_draw_handle = None
_is_baking = False
_rebuild_scheduled = False
_pending_rebuild = None  # (scene, obj_name)
# mesh_in_front 적용 전 원래 show_in_front 값 보존 {오브젝트명: bool}
_original_mesh_show_in_front: dict[str, bool] = {}


def _get_shader():
    return gpu.shader.from_builtin('UNIFORM_COLOR')


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

    if props.use_keyframes:
        _status, keyframes = _get_armature_keyframes(obj)
        # current 기준 이전/이후 키프레임 분리
        before = [f for f in keyframes if f < current]
        after  = [f for f in keyframes if f > current]
        # count=0이면 before[-0:] = 전체가 되므로 명시적 체크
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
    """해당 프레임으로 이동 후 메시 스냅샷을 GPU 배치로 반환."""
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
    """활성 오브젝트의 어니언 스킨 캐시를 증분 빌드."""
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
# GPU 드로우
# ---------------------------------------------------------------------------

def draw_onion_skins():
    """뷰포트 드로우 콜백 – 캐시된 고스트 메시 렌더링."""
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

    # 이전/이후 프레임을 가까운 순서로 정렬 (인덱스 기반 페이드용)
    before_sorted = sorted([f for f in cache if f < current], reverse=True)  # 가까운 것 먼저
    after_sorted  = sorted([f for f in cache if f > current])                 # 가까운 것 먼저

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
                # 인덱스 기반 페이드: 가장 가까운 것(i=0) → opacity 그대로
                #                      가장 먼 것(i=n-1) → 0
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
# 핸들러
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
# Timer 기반 rebuild
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
    # bpy.context.view_layer 대신 bpy.data.objects 사용 → 타이머 컨텍스트에서 안전
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


def _update_mesh_in_front(self, context):
    """메쉬 앞에 표시 설정 변경 시 – 실제 오브젝트 속성에 적용."""
    obj = _get_target_mesh(context)
    if obj:
        if self.mesh_in_front:
            # 아직 저장 안 된 경우에만 원래 값 보존
            if obj.name not in _original_mesh_show_in_front:
                _original_mesh_show_in_front[obj.name] = obj.show_in_front
            obj.show_in_front = True
        else:
            # 원래 값 복원 (저장된 게 없으면 False)
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
        description="프레임 간격 (키프레임 모드 비활성 시)",
        update=_update_cache,
    )
    use_keyframes: BoolProperty(
        name="키프레임만", default=False,
        description="아마추어 키프레임 위치에만 고스트 표시",
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
    ghost_in_front: BoolProperty(
        name="고스트 앞에 표시", default=False,
        description="어니언 스킨 고스트를 메쉬 앞에 항상 표시",
        update=_update_display,
    )
    mesh_in_front: BoolProperty(
        name="메쉬 앞에 표시", default=False,
        description="메쉬 오브젝트 자체를 항상 앞에 표시 (오브젝트 show_in_front 속성)",
        update=_update_mesh_in_front,
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
        props.enabled = not props.enabled  # _update_enabled 콜백이 처리
        return {'FINISHED'}


class MESH_OT_onion_skin_update(Operator):
    bl_idname = "mesh.onion_skin_update"
    bl_label = "어니언 스킨 갱신"
    bl_description = "어니언 스킨 캐시 강제 갱신"

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
        if props.use_keyframes:
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
        box.prop(props, "ghost_in_front")
        box.prop(props, "mesh_in_front")
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
