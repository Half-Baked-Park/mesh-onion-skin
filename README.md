# Mesh Onion Skin

Onion skin addon for Blender 5.0+ — see previous and next poses as ghost overlays while animating.

> **[한국어 README](README_KR.md)**

|Mesh|Wireframe|
|:-:|:-:|
|![Mesh Mode](images/fade_mode.png)|![Wireframe Mode](images/wireframe_mode.png)|

## Features

- **Fast GPU Rendering** — Ghosts are rendered directly on the GPU, so animation playback stays smooth
- **Keyframe Mode** — Toggle on to show ghosts only at keyframe positions; toggle off to show them at regular frame intervals
- **Smart Caching** — Only recalculates frames that actually changed, keeping scrubbing fast
- **Fade** — Ghosts further from the current frame become more transparent, with adjustable falloff
- **Wireframe Mode** — Show ghosts as outlines instead of solid shapes, so they don't block your current pose
- **In-Front Display** — Draw ghosts (and/or the mesh) on top of everything else in the scene
- **Before / After Colors** — Set different colors for past and future ghosts
- **Blender 5.0 Support** — Works with the new Layered Action system and older versions

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

1. Select a **Mesh** or its parent **Armature**
2. Open the **Onion Skin** sidebar tab and check **Enable**
3. Play the animation or scrub the timeline — ghosts appear automatically

### Panel Options

| Option | Description |
|--------|-------------|
| **Before / After** | How many ghost frames to show before and after the current frame |
| **Keyframes Only** | Show ghosts only at keyframe positions |
| **Step** | Frame interval between ghosts (when Keyframes Only is off) |
| **Opacity** | How transparent the ghosts are |
| **Fade** | Make ghosts further from the current frame more transparent |
| **Fade Falloff** | How quickly the fade effect drops off |
| **Ghost In Front** | Draw ghosts on top of all other objects |
| **Mesh In Front** | Draw the mesh on top of all other objects |
| **Wireframe** | Show ghosts as outlines instead of solid |
| **Before / After Color** | Color for past and future ghosts |

## License

MIT
