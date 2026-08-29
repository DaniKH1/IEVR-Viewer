# IEVR Viewer

IEVR Viewer is a Windows application for browsing Level-5 G4TX textures and G4MD models. It can preview G4MD geometry on the GPU, match G4TX texture variants, export selected texture maps to PNG and export skinned G4MD models to FBX without requiring Blender.

![IEVR Viewer icon](app_icon.png)

## Highlights

- Recursively browse G4TX and G4MD folders while preserving their folder structure.
- Preview every texture inside a G4TX container and export selected maps as PNG.
- Preview G4MD models in **Wireframe**, **Solid** and **Textured** modes.
- Switch between matching G4TX texture variants from the model tab.
- Export separate meshes, UVs, G4SK armature data and vertex groups to native binary FBX.
- Choose whether to export main meshes only or include LOD meshes.
- Extract every map from the G4TX variant currently selected in the 3D viewer.
- Remember G4TX and G4MD directories in `IEVR Viewer.settings.json`, stored beside the executable.

## Requirements

- Windows 10/11
- Python 3.11 or later (Python 3.13 is recommended)
- A graphics driver compatible with OpenGL 3.3 for the GPU viewer

Install the build dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run from source

```powershell
python .\g4tx_extractor.py
```

Choose **G4TX Folder…** and/or **G4MD Folder…** from the toolbar. Both folders may be loaded independently. When their internal folder layout matches, the model viewer automatically finds G4TX containers in the corresponding location.

## Build your own executable

From PowerShell in this repository:

```powershell
.\build_exe.ps1
```

The build creates:

```text
dist\IEVR Viewer.exe
```

Copy `IEVR Viewer.exe` wherever you want to use it. At first launch the application creates `IEVR Viewer.settings.json` next to the executable. Keep that JSON file beside the executable to restore the last selected folders automatically.

## Using the application

### G4TX Textures

1. Choose **G4TX Folder…**.
2. Expand a folder in the tree and select a `.g4tx` file.
3. Tick the texture maps to export.
4. Select **Export Selected to PNG…**.

### G4MD Models

1. Choose **G4MD Folder…**.
2. Select a `.g4md` file from the tree.
3. Use **Wireframe**, **Solid** or **Textured** to select its display mode.
4. Drag with the left mouse button to orbit. Use the mouse wheel to zoom.
5. Use the arrow buttons to select a matching G4TX texture variant.
6. Choose **Main meshes only** or **Include LOD meshes**, then select **Export to FBX…**.

The generated FBX keeps each source mesh separate, includes UVs, the matching G4SK armature and vertex groups, is scaled to 1.00 in Blender and is converted to Blender's Z-up orientation. The texture folder opens automatically after export.

## Repository layout

```text
g4tx_extractor.py  Application source
build_exe.ps1      Windows PyInstaller build script
requirements.txt   Python build dependencies
app_icon.png       Application icon used inside the interface
app_icon.ico       Windows executable icon
```

## Credits

- Idea Designer: **DaniKH**
- Original Blender add-on author: **The-RealBobi** - [G4_Blender](https://github.com/The-RealBobi/G4_Blender)
- Built with Codex

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution information.
