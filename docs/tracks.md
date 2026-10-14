# Elevated oval track

Run from the repository root:

```bash
uv run warptracer demo oval
```

Open **`outputs/oval.html`** for the replay. The 120-second run uses disparity
navigation, lean physics, 108-beam LiDAR at 60 Hz, and graph/fused execution.
It chooses CUDA when available, otherwise CPU. The JSON reports lap progress
and barrier-contact substeps; the NPZ retains poses, controls and scans.

## Geometry

| Setting | Default |
| --- | --- |
| Outer road dimensions | 16 × 10 m |
| Center ellipse axes | 7 × 4 m |
| Lane | Nominal 2 m; inner/outer ellipse axes differ by 2 m |
| Elevation | Smooth rise and fall from 0 to 0.4 m |
| Maximum surface grade | About 7.9% (4.5°) |
| Barriers | 1 m high, 0.12 m thick, following both road edges |
| Rendered objects | Box car, one road mesh, two barrier meshes |

The elevation varies smoothly along world X, including gentle cross slopes
around the ends. There are no textures, wheel meshes or external assets.
Barriers are tall enough to intercept the tilted scanner's rays over the hills.
The shared static mesh has 6,144 triangles, with 192 segments around the loop
and eight strips across the lane. Playback and LiDAR use identical vertices and
faces. Lean physics evaluates the matching analytic surface; tests check mesh
height against it within 1 mm.
`uv run warptracer demo oval-grid` keeps this road and physics but replaces
sensor geometry with EDT clearance fields and sampled terrain height. See the
[grid LiDAR experiment](grid-lidar.md) for its accuracy and timing comparison.

## Driving behavior

The chassis follows the road height and normal, so pitch/roll and the mounted
scanner change as it travels. Reported world velocity includes vertical motion.
The spawn is on the centerline, pointing counterclockwise. Both barriers test
the padded chassis footprint and stop motion on contact.

Navigation still selects targets from disparity-filtered LiDAR. It disables
the room-specific corner latch and uses a bounded lookahead to convert the
selected ray direction to bicycle steering. It does not read a centerline or
map. Default tests complete multiple laps at 30, 40 and 60 Hz with zero contacts.

Lap counts estimate net counterclockwise angular progress from recorded poses;
headless demos omit them. They are useful demo diagnostics, not an RL reward or
continuous device-side timing system.

The lean tire/acceleration model remains planar. Terrain following constrains
the chassis to the road; suspension, airborne motion and gravity along slopes
are future physics work. Newton oval support is not implemented.

## Benchmark or customize

On a GPU runtime:

```bash
uv run warptracer benchmark oval
```

This measures navigation with 1, 64 and 256 independent cars and writes
`outputs/oval-benchmark.json`. The `grid` preset compares sensing backends on the
same track; the other presets use the room. See the
[benchmark guide](benchmarking.md) for timing and units. GPU throughput for the
grid backend still needs measurement. The user's mesh-oval Tesla T4 run measured
6,490 / 372,202 / 919,642 aggregate navigation transitions/s for 1 / 64 / 256
cars; this does not predict grid performance.

For geometry experiments, configure it in Python:

```python
from warptracer.terrain import OvalTrack
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation

track = OvalTrack(elevation=0.2)
sim = Simulation(track=track, scenario="drive", engine="lean",
                 lidar=LidarConfig(frequency=60))
```

Set `elevation=0` for a flat loop. Dimensions and grade are validated, but changing
geometry or controller settings requires another navigation check.
