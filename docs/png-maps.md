# Train on a PNG map

Use an occupancy image: **white is drivable; black, gray and transparent areas
are obstacles**. A PNG has no physical scale, so you must provide meters per
pixel. For example, 200 pixels across a known 10 m distance means `0.05` m/pixel.
This importer creates a **flat** track with 1 m walls. It does not infer elevation
from pixel brightness.

## Local workflow

Put your image at `maps/my-track.png`, then prepare it once:

```bash
uv run warptracer map maps/my-track.png --resolution 0.05
```

This creates `maps/my-track.json`, checks the route/car clearance and writes
`outputs/maps/my-track.png`. Open the preview: green is the route, orange shows
its initial direction. Default automatic extraction expects one closed racing
loop. Free space must be surrounded by an occupied/unknown border. A room, maze,
junction or multiple loops needs an explicit route (below).

Train with the usual preset and one map option:

```bash
uv run warptracer train oval --map maps/my-track.png
uv run warptracer evaluate --view
```

Here `oval` selects the existing 1,024-car **training settings**; `--map` replaces
the oval geometry. `large` selects 4,096 cars; `smoke` checks the path on CPU.
Both `.png` and `.json` paths work with `--map`. The PNG path uses its same-stem
JSON sidecar. The output directory is still `outputs/ppo`; use a different
`--output` directory if you want to keep another experiment's checkpoints.

The checkpoint embeds the occupancy bitmap, route, scale and origin. Evaluation
and resume restore that geometry even if the original PNG/JSON files are moved.
Existing oval checkpoints remain supported. Resume cannot switch to a new map.

## Colab

Open [the PPO notebook](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/ppo_colab.ipynb).
After setup, set `use_png = True`, set `meters_per_pixel`, and run the map cell.
Upload one PNG when prompted. The cell creates the sidecar, validates it and
displays the route preview. Continue with the normal training/evaluation cells.
Leave `use_png = False` for the original elevated oval. Resuming restores the
checkpoint map without requiring another upload; leave `use_png = False` in
that case.

## Map settings

The generated sidecar is deliberately small:

```json
{
  "image": "my-track.png",
  "resolution": 0.05
}
```

Paths are relative to the sidecar. Optional settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `origin` | `[0.0, 0.0]` | World XY of the image's bottom-left pixel corner, in meters; no yaw rotation |
| `free_threshold` | `0.95` | Minimum grayscale brightness in [0,1] to count as free; e.g. ROS-style unknown gray 205 stays blocked |
| `negate` | `false` | Invert brightness for black-free/white-occupied images; transparent pixels remain blocked |
| `wall_height` | `1.0` | Uniform obstacle height in meters |
| `reverse` | `false` | Reverse the route direction; inferred routes otherwise run counterclockwise |
| `centerline` | automatic | CSV with ordered world-space `x,y` coordinates, no header; closes from last point to first |

For ambiguous images, draw/supply an ordered route and set, for example,
`"centerline": "my-track-centerline.csv"`. Include enough points to follow every
corner. The importer resamples it at approximately 0.1 m and checks the whole
loop's clearance. It never silently guesses a branch through a junction.
Colors are converted to grayscale; decorative racing photos or colored track
diagrams should first be converted into an occupancy mask.

## Runtime behavior and limits

- PNG pixels become one shared EDT geometry used by all cars; preprocessing and
  route extraction run once on CPU. Full 3D LiDAR rays run in Warp against flat
  ground and finite-height walls. No sensor triangle mesh is constructed.
- Map collision uses a conservative circle enclosing the 0.52 x 0.26 m chassis
  (radius about 0.291 m), plus distance-field discretization margins. It checks
  the swept motion at every 240 Hz substep. Narrow lanes may be rejected even
  where a precisely aligned rectangle could fit.
- Cars spawn along the validated route with the existing small heading variation.
  Reward is signed continuous route-distance progress, with the existing failure,
  lap and timeout rules. Local segment projection preserves route order through
  nearby hairpins. A full forward route length completes a lap.
- PNGs are limited to four million pixels. Resolution controls physical scale,
  not resampling: resizing the image requires updating meters per pixel. Coarse
  cells produce coarser LiDAR surfaces and more conservative collisions.
- Warp playback includes a flat road and extruded wall outlines derived
  from the sensor field. Rendering meshes are created only for replay.
- This adds occupancy maps, not height-map import, SLAM, bridges, map randomization
  or interacting cars. GPU throughput and learning on a new map still need testing
  on your CUDA machine/Colab.
