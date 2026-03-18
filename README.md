# Mesh Onion Skin

Onion skin addon for Blender 5.0+ — see previous and next poses as ghost overlays while animating.

> **[한국어 README](README_KR.md)**

|All Frames|Keyframe Only|Wireframe|
|:-:|:-:|:-:|
|![All Frames](images/01_all_frame.png)|![Keyframe Only](images/02_keyframe_only.png)|![Wireframe](images/05_wireframe.png)|

|Active Mode|Scene Mode|Collection Mode|
|:-:|:-:|:-:|
|![Active](images/07_activemode.png)|![Scene](images/08_scenemode.png)|![Collection](images/09_collectionmode.png)|

## Features

- **Three Target Modes** — Active (single object), Scene (all animated meshes), or Collection (specific group)
- **Fast GPU Rendering** — Ghosts are rendered directly on the GPU, so animation playback stays smooth
- **Keyframe Mode** — Show ghosts only at keyframe positions (Active mode only)
- **Smart Caching** — Only recalculates frames that actually changed, keeping scrubbing fast
- **Frame-First Baking** — In multi-object modes, minimizes frame jumps by baking all objects per frame in a single pass
- **Fade** — Ghosts further from the current frame become more transparent, with adjustable falloff
- **Wireframe Mode** — Show ghosts as outlines instead of solid shapes, so they don't block your current pose
- **In-Front Display** — Choose to draw ghosts or the mesh on top of everything else (mutually exclusive)
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
3. Choose a **Mode** from the dropdown:
   - **Active** — Ghosts for the selected object only
   - **Scene** — Ghosts for all visible animated meshes in the scene
   - **Collection** — Ghosts for all visible animated meshes in a specific collection
4. Play the animation or scrub the timeline — ghosts appear automatically

> **Collection mode note:** Make sure armatures and their child meshes are in the **same collection**. If a mesh is in Collection A but its armature is in Collection B, the ghost will appear when filtering Collection A (where the mesh lives), not Collection B.

### Panel Options

| Option | Description |
|--------|-------------|
| **Mode** | Target mode — Active, Scene, or Collection |
| **Collection** | Which collection to use (Collection mode only) |
| **Max Objects** | Maximum number of objects to process (Scene/Collection modes) |
| **Before / After** | How many ghost frames to show before and after the current frame |
| **Keyframes Only** | Show ghosts only at keyframe positions (Active mode only) |
| **Step** | Frame interval between ghosts |
| **Opacity** | How transparent the ghosts are |
| **Fade** | Make ghosts further from the current frame more transparent |
| **Fade Falloff** | How quickly the fade effect drops off |
| **In Front** | None / Ghost (draw ghosts on top) / Mesh (draw mesh on top) |
| **Wireframe** | Show ghosts as outlines instead of solid |
| **Before / After Color** | Color for past and future ghosts |

## License

MIT
