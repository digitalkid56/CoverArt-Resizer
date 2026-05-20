# Cartridge Stamper

UI/UX prototype for an AYN Thor-optimized cover-art stamping workflow.

## Goals
- Load a template from a designated template-assets directory.
- Import cover art from a designated folder or pick one image manually.
- Apply default placement automatically with optional spacing/scale tuning.
- Export stamped output as PNG to default output folder or user-selected destination.

## Current status
This repository includes a v1 Jetpack Compose workflow for the AYN Thor:
- **Template source**: bundled app assets from `app/src/main/assets/templates`.
- **Template picker**: dropdown selection with automatic console/system pairing and conditional template options.
- **Cover art source**: dropdown between `/AYN Thor/ES-DE/downloaded_media/<system>/physicalmedia`, `/AYN Thor/ES-DE/downloaded_media/<system>/covers`, and a specific folder picker.
- **Preview**: cover art composited under the selected PNG template, clipped 5 px inside the cartridge silhouette, with scale and offset controls.
- **Export**: PNG output to `/AYN Thor/Download/ICONS/<system>/<image>.png`, with an alternate "Choose Destination" export path.

## Template system mapping
- Game Boy -> `gb`
- Game Boy Advance -> `gba` when a GBA/Advance template is added
- Game Boy Color -> `gbc`
- Nintendo DS -> `nds`
- Nintendo 3DS -> `n3ds`
- Nintendo GameCube -> `gc`
- PSP UMD -> `psp`
- Nintendo Switch -> `switch`

## Next implementation milestones
1. Tune per-template default placement metadata for each cartridge type.
2. Add batch export from the selected physical-media folder.
3. Add saved presets for repeated platform-specific stamping runs.
