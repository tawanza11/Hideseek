# Hide & Seek 3D Replay

This is a spectator view of a **recorded match played by saved models**. The 3D scene does not change the game or the agent observations. It shows both agents even when the seeker cannot see the hider; the yellow sight line appears only when the game reports visibility.

## Open the included match

From `viewer3d/`:

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. Drag to rotate, scroll to zoom, press Play, or move the timeline. The included `public/replay-example.json` is exported from saved V6 curriculum Seeker and V3 Hider models on V5F001 with training seed 41 and episode seed 800000.

## Export another real match

From the repository root, with the Python dependencies installed:

```bash
source .venv/bin/activate
python -m evaluation.export_replay --model v6-curriculum --map V5F001 --training-seed 41 --episode-seed 800000 --output viewer3d/public/replay-example.json
```

Options for `--model`: `v4`, `v5`, `v6-flat`, `v6-curriculum`. Options for `--map`: `default`, V5E001–V5E010, or V5F001–V5F020. To inspect without replacing the included match, export to another path and open its JSON using the viewer's upload button.

## Replay format

`schema_version: 1`, `map_rows` (`#` walls, `.` floor), `frames`, and `outcome` are the main fields. Each frame has `step`, `seeker: [x,y]`, `hider: [x,y]`, and `visible` (whether the **seeker** can see the hider). Coordinates start at the top left. Frames may have `action` and `event`. `blocks`, `ramps`, and `low_walls` are arrays of `[x,y]` cells. They can be placed at the root for static objects or on individual frames for changing objects. The V6 export has empty arrays because those mechanics do not exist in that game yet. `outcome` is `seeker_captured` or `hider_survived`.

The viewer only displays object positions from JSON. A future environment must supply its own real object rules and observations before a model can learn to use blocks or ramps.
