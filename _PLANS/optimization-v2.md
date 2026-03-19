# Mesh Onion Skin 알고리즘 최적화 플랜

> 목표: max_objects 50 → 500+ 확장, 뷰포트 프리징 제거
> 방식: C++ 전환 대신 Python 알고리즘 최적화 (depsgraph가 진짜 병목이라 C++ 효과 10~15%)

---

## 핵심 발견: 즉시 수정할 버그급 이슈

`_schedule_rebuild()` (line 456)에서 `clear_cache()`를 무조건 호출 → `rebuild_cache()`의 incremental diffing (line 300-309)이 완전히 무효화됨. 이것만 고쳐도 프레임 스크러빙 시 **6x 성능 향상**.

---

## Phase 1 — 캐시 보존 + 점진적 베이킹

### 1-1. `_schedule_rebuild()`에서 `clear_cache()` 제거
- 프레임 이동 시 기존 캐시 재활용
- 1프레임 이동 시 `frame_set` 호출: 6회 → 1회
- `use_flat`/`use_keyframes` 등 포맷 변경만 full clear → `_update_cache` 콜백 분리

### 1-2. 점진적 베이킹 (Timer 기반)
- `_bake_queue`: `list[tuple[int, list]]` — (frame, [objects]) 우선순위 큐
- `_progressive_bake_tick()`: bpy.app.timers 콜백, 틱당 1-2 `frame_set` 호출
- 틱 끝에 `scene.frame_set(current)` 복원 → 뷰포트 깜빡임 방지
- `_bake_generation` 세대 카운터: 스크러빙 시 stale 작업 즉시 취소

### 1-3. 우선순위 큐
- `_build_prioritized_queue()`: 현재 프레임에 가까운 고스트부터 베이킹
- 중요한 고스트 즉시 표시, 먼 고스트는 백그라운드 채움

### 1-4. 스마트 캐시 무효화
- 프레임 이동 시 겹치는 프레임 자동 재활용 (sliding window)
  - frame 50→51: {47,48,49,51,52,53} → {48,49,50,52,53,54} → 4개 재활용, 2개만 베이킹
- `_compute_cache_fingerprint()`: 오브젝트별 fingerprint로 변경 안 된 오브젝트 스킵

### 1-5. UI 피드백
- `bake_batch_size` 프로퍼티 노출 (default 2, min 1, max 10)
- 베이킹 진행률 패널 표시

### Phase 1 예상 효과
- max_objects 50 → 200+ 가능
- 프레임 스크러빙 6x 빨라짐
- 뷰포트 프리징 제거

---

## Phase 2 — GPU 드로우 최적화

### 2-1. 셰이더 변경
- `UNIFORM_COLOR` → `SMOOTH_COLOR` (per-vertex color)
- Blender 5.0에서 빌트인 이름 확인 필요, 없으면 커스텀 GLSL 대체

### 2-2. 캐시 구조 변경
- GPUBatch 대신 raw numpy 배열 `(positions, indices)` 저장
- `_geometry_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]`

### 2-3. 배치 머징
- `_build_merged_batches()`: before/after 각각 하나의 mega-batch
- 인덱스 오프셋팅: 메시 B의 인덱스 += 메시 A의 vertex 수
- per-vertex color 배열: 같은 고스트의 모든 vertex에 동일 RGBA
- `_merged_dirty` 플래그: 캐시 변경 시만 재빌드

### 2-4. `.tolist()` 제거
- line 252, 262의 `.tolist()` → numpy 배열 직접 전달 (Blender 4.0+)

### Phase 2 예상 효과
- 드로우콜: objects × ghosts (최대 1200) → **2개** (before + after)
- 200 오브젝트에서도 60fps 드로잉 유지

### VRAM 추정 (10K vertex 메시 기준)
| 항목 | 고스트당 |
|------|---------|
| positions (float32 xyz) | 120 KB |
| color (float32 RGBA) | 160 KB |
| indices (int32 tri) | 216 KB |
| **합계** | **~496 KB** |
| 200 obj × 6 ghosts | **~581 MB** |

→ bbox 폴백 + 메모리 캡 필수 (Phase 3에서)

---

## Phase 3 — 공간/시간 컬링

### 3-1. 뷰 프러스텀 컬링
- `rv3d.perspective_matrix`에서 6개 프러스텀 평면 추출 (Gribb-Hartmann)
- `obj.bound_box` × `matrix_world`를 평면 테스트
- 카메라 밖 오브젝트 베이킹/드로잉 스킵
- 예상: 30-70% 오브젝트 제외

### 3-2. 모션 컬링
- 프레임 간 `matrix_world.translation` + 루트 본 위치 비교
- 델타 < threshold → 고스트 스킵
- `motion_threshold` 프로퍼티 (default 0.005)
- 예상: 정지 오브젝트 20-50% 절감

### 3-3. 거리 기반 LOD
- Near: 풀 고스트, 솔리드
- Mid: 고스트 수 절반, 와이어프레임 강제
- Far: before 1 + after 1만, 와이어프레임
- 예상: 총 고스트 ~40% 절감

### 3-4. 바운딩 박스 폴백
- 매우 먼 고스트: 8 vertex bbox만 렌더 (96 bytes/ghost)
- VRAM 캡 초과 시 자동 전환

### 3-5. 메모리 캡
- `max_vram_mb` 프로퍼티 (default 256 MB)
- 초과 시 가장 먼 고스트부터 제거
- 현재 VRAM 사용량 UI 표시

### Phase 3 예상 효과
- max_objects 500+ 가능
- 실제 연산량 추가 50-70% 감소

---

## 주요 리스크

| 리스크 | 대응 |
|--------|------|
| VRAM 폭발 (200obj × 6ghost × 10K = 580MB+) | bbox 폴백 + 메모리 캡 + LOD |
| `SMOOTH_COLOR` Blender 5.0 호환 | 확인 후 없으면 커스텀 GLSL 3줄 |
| 재생 중 Timer 충돌 | `is_animation_playing` 체크, budget 비활성화 |
| 오브젝트 삭제/이름변경 | 큐에 이름 저장, `bpy.data.objects.get()` null 가드 |
| 부분 표시 중 시각적 어색함 | 우선순위 큐 + 진행률 UI |
| Lambda 캡처 문제 | `_bake_generation` 값 캡처 (functools.partial) |
| 다중 뷰포트 프러스텀 | 활성 뷰포트만 사용 (단순화) |

---

## 수정 대상 함수 목록

### 수정
| 함수 | 위치 | 변경 내용 |
|------|------|----------|
| `_get_shader()` | line 39 | `SMOOTH_COLOR` 반환 (Phase 2) |
| `_bake_mesh_snapshot()` | line 223 | raw numpy 반환 + `.tolist()` 제거 |
| `rebuild_cache()` | line 268 | 오케스트레이터로 전환, 큐 생성만 |
| `draw_onion_skins()` | line 337 | merged batch 2개만 draw |
| `clear_cache()` | line 120 | merged batch + geometry cache도 클리어 |
| `_schedule_rebuild()` | line 454 | `clear_cache()` 제거 |
| `_update_cache()` | line 504 | incremental / full 분리 |
| `_update_display()` | line 530 | `_merged_dirty = True` 추가 |
| `MeshOnionSkinProps` | line 552 | max_objects max=500, 새 프로퍼티 추가 |

### 신규
| 함수 | 용도 |
|------|------|
| `_progressive_bake_tick()` | 타이머 콜백, 틱당 N 프레임 베이킹 |
| `_build_prioritized_queue()` | 현재 프레임 기준 우선순위 정렬 |
| `_compute_cache_fingerprint()` | 오브젝트별 변경 감지 |
| `_build_merged_batches()` | before/after mega-batch 빌드 |
| `_is_in_view_frustum()` | 프러스텀 컬링 테스트 |
| `_compute_lod_level()` | 거리 기반 LOD 결정 |
| `_estimate_vram()` | VRAM 사용량 계산 |

---

## 구현 순서

```
Phase 1 (캐시 + 점진적 베이킹) ← 최우선
  ├─ 1-1. clear_cache 제거 (즉시)
  ├─ 1-2. 점진적 베이킹 Timer
  ├─ 1-3. 우선순위 큐
  ├─ 1-4. fingerprint 캐시 무효화
  └─ 1-5. UI 피드백 + max_objects 상향

Phase 2 (GPU 드로우)
  ├─ 2-1. SMOOTH_COLOR 셰이더
  ├─ 2-2. raw numpy 캐시
  ├─ 2-3. 배치 머징
  └─ 2-4. .tolist() 제거

Phase 3 (컬링 + LOD)
  ├─ 3-1. 프러스텀 컬링 ✅
  ├─ 3-2. 모션 컬링
  ├─ 3-3. 거리 LOD
  ├─ 3-4. bbox 폴백
  └─ 3-5. 메모리 캡

Phase 4 (하이폴리/모디파이어 최적화) — 미구현
  ├─ 4-1. Decimation 프록시
  ├─ 4-2. to_mesh 캐싱
  ├─ 4-3. 모디파이어 스킵
  └─ 4-4. 적응형 고스트 수
```

---

## Phase 4 — 하이폴리 / 모디파이어 / 지오메트리노드 최적화

> Phase 1-3은 "오브젝트 수가 많은" 시나리오 최적화.
> Phase 4는 "오브젝트 수는 적지만 개별 오브젝트가 무거운" 시나리오 최적화.
> 병목: `eval_obj.to_mesh()` — 모디파이어 스택/GeoNode 전체를 재평가하므로 오브젝트 1개에 수백ms 소요 가능.

### 4-1. Decimation 프록시 베이킹

**문제**: 500K vertex 메시의 고스트 6개 = 3M vertex → VRAM 폭발 + 느린 to_mesh

**해결**: 고스트용 간소화 메시 생성
- 베이킹 시 임시 Decimate 모디파이어 추가 (ratio=0.1~0.3)
- 또는 bmesh.ops.dissolve_degenerate 사용
- `ghost_lod_ratio` 프로퍼티로 조절 (default 0.2 = 원본의 20%)

**적용 시점**: `_bake_mesh_snapshot` 내부
```
# 의사코드
if props.use_ghost_lod and vertex_count > props.lod_vertex_threshold:
    # 임시 Decimate 모디파이어 추가
    mod = obj.modifiers.new("_onion_decimate", 'DECIMATE')
    mod.ratio = props.ghost_lod_ratio
    # evaluated mesh 재취득
    depsgraph.update()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    # 베이킹 후 모디파이어 제거
    obj.modifiers.remove(mod)
```

**리스크**:
- Decimate 모디파이어 추가/제거가 depsgraph를 오염시킬 수 있음
- 해결: evaluated copy에서 작업하거나, 별도 빈 오브젝트에 데이터를 복사해서 decimation

**예상 효과**: 500K → 100K vertex per ghost. VRAM 80% 감소, to_mesh 자체는 여전히 비쌈

### 4-2. to_mesh 결과 캐싱 (Cross-frame 재활용)

**문제**: 매 고스트 프레임마다 `to_mesh()` 호출 = 모디파이어 전체 재평가

**해결**: 이전 프레임 대비 변화량 체크 → 변화 없으면 이전 결과 재활용
- `frame_set` 후 `obj.matrix_world` + 주요 본 위치 비교
- 변화 미미하면 기존 캐시 엔트리 복사 (to_mesh 스킵)

**데이터 구조**:
```
_motion_hash: dict[str, dict[int, tuple[float, ...]]] = {}
# obj_name -> {frame -> (tx, ty, tz, bone0_x, bone0_y, ...)}
```

**적용 시점**: `_progressive_bake_tick` 내, `_bake_mesh_snapshot` 호출 전
```
# 의사코드
current_hash = _compute_motion_hash(obj, scene)
prev_frame_hash = _motion_hash.get(obj.name, {}).get(prev_frame)
if prev_frame_hash and _hash_distance(current_hash, prev_frame_hash) < threshold:
    # 이전 프레임 캐시 재활용
    _onion_cache[obj.name][frame] = _onion_cache[obj.name][prev_frame]
    continue  # to_mesh 호출 스킵
```

**리스크**:
- 본 위치만 비교하면 셰이프키/cloth/soft body 변형 감지 못함
- 해시 계산 자체가 frame_set 후에만 가능 (to_mesh 대비 비용은 미미)

**예상 효과**: 정지 구간에서 to_mesh 호출 90%+ 감소

### 4-3. 모디파이어 선택적 비활성화

**문제**: Subdivision Surface, Smooth, Bevel 등 형상에 영향 적은 모디파이어도 고스트 베이킹 시 재평가됨

**해결**: 고스트 베이킹 시 불필요 모디파이어 임시 비활성화
- `SUBSURF`, `SMOOTH`, `BEVEL`, `SOLIDIFY` 등 → show_viewport 임시 off
- `ARMATURE`, `MESH_DEFORM` 등 변형 모디파이어는 유지
- `ghost_skip_modifiers` 프로퍼티: 스킵할 모디파이어 타입 리스트

**적용 시점**: `_bake_mesh_snapshot` 전후
```
# 의사코드
SKIPPABLE = {'SUBSURF', 'SMOOTH', 'BEVEL', 'SOLIDIFY', 'WEIGHTED_NORMAL'}
disabled = []
for mod in obj.modifiers:
    if mod.type in SKIPPABLE and mod.show_viewport:
        mod.show_viewport = False
        disabled.append(mod)
# ... bake ...
for mod in disabled:
    mod.show_viewport = True
```

**리스크**:
- 모디파이어 비활성화가 최종 형상을 변형 → 고스트가 원본과 다르게 보일 수 있음
- 사용자 선택에 맡기기 (opt-in)
- depsgraph 오염: show_viewport 변경이 다른 오브젝트에 영향 줄 수 있음

**예상 효과**: Subdivision Level 2 기준 vertex 4x 감소 → to_mesh 4x 빨라짐

### 4-4. 적응형 고스트 수

**문제**: 하이폴리 오브젝트에 before 3 + after 3 = 6 고스트 → to_mesh 6회

**해결**: vertex 수 기반 자동 고스트 수 조절
- `auto_ghost_count` 모드: vertex > 100K → before/after 각 1개, > 50K → 각 2개
- 또는 총 고스트 vertex 예산 (budget) 기반
- `ghost_vertex_budget` 프로퍼티 (default 1M vertices)

**적용 시점**: `_get_target_frames` 내
```
# 의사코드
if props.auto_ghost_count:
    vertex_count = len(obj.data.vertices)  # base mesh vertex count
    if vertex_count > 100_000:
        effective_before = min(1, props.count_before)
        effective_after = min(1, props.count_after)
    elif vertex_count > 50_000:
        effective_before = min(2, props.count_before)
        effective_after = min(2, props.count_after)
```

**리스크**: 없음 (단순한 조건부 축소)

**예상 효과**: 하이폴리 오브젝트의 to_mesh 호출 50-70% 감소

### Phase 4 구현 우선순위

| 순위 | 항목 | 복잡도 | 효과 | 리스크 |
|------|------|--------|------|--------|
| 1 | 4-4. 적응형 고스트 수 | 낮음 | 중간 | 없음 |
| 2 | 4-2. to_mesh 캐싱 | 중간 | 높음 | 변형 미감지 가능 |
| 3 | 4-3. 모디파이어 스킵 | 중간 | 높음 | 형상 차이 |
| 4 | 4-1. Decimation 프록시 | 높음 | 매우높음 | depsgraph 오염 |

### Phase 4 새 프로퍼티

```
auto_ghost_count: BoolProperty (default False)
ghost_vertex_budget: IntProperty (default 1_000_000)
ghost_lod_ratio: FloatProperty (default 0.2, min 0.01, max 1.0)
lod_vertex_threshold: IntProperty (default 50_000)
skip_heavy_modifiers: BoolProperty (default False)
```
