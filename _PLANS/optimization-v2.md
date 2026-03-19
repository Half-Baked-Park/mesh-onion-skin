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
  ├─ 3-1. 프러스텀 컬링
  ├─ 3-2. 모션 컬링
  ├─ 3-3. 거리 LOD
  ├─ 3-4. bbox 폴백
  └─ 3-5. 메모리 캡
```
