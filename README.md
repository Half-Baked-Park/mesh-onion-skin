# Mesh Onion Skin

A GPU-based onion skin addon for 3D mesh animations in Blender 5.0+.

> **[한국어 README](README_KR.md)**

![Fade Mode](images/fade_mode.png)

## Features

- **GPU-Accelerated Rendering** — Ghost meshes are drawn entirely on the GPU via `gpu` module batches, keeping CPU overhead near zero
- **Keyframe-Aware Mode** — Ghosts snap to actual armature keyframes instead of fixed intervals, so you see exactly where your poses land
- **Incremental Cache** — Only changed frames are re-baked; scrubbing the timeline stays responsive even with complex meshes
- **Fade Falloff** — Ghost opacity decreases with temporal distance from the current frame, with adjustable curve strength
- **Wireframe Mode** — Display ghosts as wireframe edges for a cleaner view without occluding the current pose
- **In-Front Options** — Independent depth override for ghosts and the mesh itself, so ghosts can render above scene geometry
- **Before / After Colors** — Separate color settings for past and future ghosts
- **Blender 5.0 Layered Action Support** — Full support for the new Layered Action system with legacy fallback

|Keyframe Mode|Wireframe Mode|
|:-:|:-:|
|![Keyframe Mode](images/keyframe_mode.png)|![Wireframe Mode](images/wireframe_mode.png)|

## Requirements

- Blender **5.0** or later

## Installation

1. Download the `.py` file for your preferred language:
   | File | Language |
   |------|----------|
   | `mesh_onion_skin_en.py` | English |
   | `mesh_onion_skin_kr.py` | 한국어 |

2. In Blender: **Edit > Preferences > Add-ons > Install**
3. Select the downloaded file and enable the addon
4. The panel appears in **View3D > Sidebar (N) > Onion Skin** tab

## Usage

1. Select a **Mesh** object or its parent **Armature**
2. Open the **Onion Skin** sidebar tab and check **Enable**
3. Play the animation or scrub the timeline — ghosts update automatically

### Panel Options

| Option | Description |
|--------|-------------|
| **Before / After** | Number of ghost frames before and after the current frame |
| **Keyframes Only** | Snap ghosts to armature keyframe positions only |
| **Step** | Frame interval between ghosts (when Keyframes Only is off) |
| **Opacity** | Overall ghost transparency |
| **Fade** | Enable opacity falloff by temporal distance |
| **Fade Falloff** | Curve strength of the fade effect |
| **Ghost In Front** | Draw ghosts above all scene geometry |
| **Mesh In Front** | Set the object's `show_in_front` property |
| **Wireframe** | Render ghosts as wireframe edges |
| **Before / After Color** | Color for past / future ghost groups |

## License

MIT
