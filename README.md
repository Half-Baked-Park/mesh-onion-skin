# Mesh Onion Skin — Blender Addon

A GPU-based onion skin addon for 3D mesh animations in Blender 5.0+.

GPU 기반 3D 메시 애니메이션 어니언 스킨 Blender 5.0+ 애드온.

---

## Files / 파일

| File | Description |
|------|-------------|
| `mesh_onion_skin_en.py` | English version |
| `mesh_onion_skin_kr.py` | 한국어 버전 |

---

## Features / 기능

- **GPU-accelerated ghost rendering** — zero overhead on the CPU render path
  **GPU 가속 고스트 렌더링** — CPU 렌더 경로에 오버헤드 없음
- **Keyframe-aware mode** — ghosts snap to actual armature keyframes
  **키프레임 인식 모드** — 아마추어 실제 키프레임 위치에 고스트 표시
- **Incremental cache** — only re-bakes frames that changed
  **증분 캐시** — 변경된 프레임만 다시 베이크
- **Fade falloff** — opacity decreases with temporal distance
  **페이드 감쇠** — 시간적 거리에 따라 불투명도 감소
- **Wireframe mode** — display ghosts as edges instead of solid
  **와이어프레임 모드** — 솔리드 대신 엣지로 고스트 표시
- **Ghost / Mesh in-front options** — independent depth override controls
  **고스트 / 메시 앞에 표시 옵션** — 독립적인 깊이 오버라이드 컨트롤
- **Blender 5.0 Layered Action support** with legacy fallback
  **Blender 5.0 레이어드 액션 지원** + 레거시 폴백

---

## Requirements / 요구 사항

- Blender **5.0** or later (uses `gpu.shader.from_builtin('UNIFORM_COLOR')` and Layered Action API)
- Blender **5.0** 이상 (`gpu.shader.from_builtin('UNIFORM_COLOR')` 및 레이어드 액션 API 사용)

---

## Installation / 설치

1. Download the `.py` file for your preferred language.
   선호하는 언어의 `.py` 파일을 다운로드하세요.

2. In Blender: **Edit → Preferences → Add-ons → Install**
   Blender에서: **편집 → 환경설정 → 애드온 → 설치**

3. Select the downloaded file and enable the addon.
   다운로드한 파일을 선택하고 애드온을 활성화하세요.

4. The panel appears in **View3D → Sidebar (N) → Onion Skin** tab.
   패널은 **뷰3D → 사이드바(N) → Onion Skin** 탭에 나타납니다.

---

## Usage / 사용법

1. Select a **Mesh** or its parent **Armature**.
   **메시** 또는 부모 **아마추어**를 선택하세요.

2. Open the **Onion Skin** sidebar tab and check the **Enable** checkbox (or press the Enable button).
   **Onion Skin** 사이드바 탭을 열고 **활성화** 체크박스를 체크하세요 (또는 활성화 버튼 클릭).

3. Play the animation or scrub the timeline — ghosts update automatically.
   애니메이션을 재생하거나 타임라인을 스크럽하면 고스트가 자동으로 업데이트됩니다.

### Panel options / 패널 옵션

| Option | Description | 설명 |
|--------|-------------|------|
| Before / After | Ghost count before/after current frame | 현재 프레임 이전/이후 고스트 수 |
| Keyframes Only | Snap ghosts to armature keyframe positions | 아마추어 키프레임 위치에만 고스트 표시 |
| Step | Frame interval (when Keyframes Only is off) | 프레임 간격 (키프레임 모드 비활성 시) |
| Opacity | Overall ghost transparency | 전체 고스트 투명도 |
| Fade | Fade opacity by temporal distance | 시간적 거리에 따른 페이드 |
| Fade Falloff | Curve strength of the fade | 페이드 커브 강도 |
| Ghost In Front | Draw ghosts above all geometry | 모든 지오메트리 위에 고스트 그리기 |
| Mesh In Front | Set object show_in_front property | 오브젝트 show_in_front 속성 설정 |
| Wireframe | Render ghosts as edges | 고스트를 엣지로 렌더링 |
| Before / After Color | Color per ghost group | 그룹별 고스트 색상 |

---

## License / 라이선스

MIT
