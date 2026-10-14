# EDT distance-field LiDAR

The optional `grid` backend now uses **EDT signed-distance fields and Warp ray
marching**. The previous cell/tile traversal implementation has been deleted.
It casts full 3D rays without building a sensor triangle mesh or BVH.
The existing mesh backend stays the default. Try the grid replay with:

```bash
uv run warptracer demo oval-grid
```

Open `outputs/oval-grid.html`. This runs the same 120-second oval navigation
task with 108 beams at 60 Hz. Vehicle physics and the displayed road remain the
same; sensing uses a rasterized approximation of the track.

## Compare on Colab

Select a GPU runtime, run the setup cell in the
[benchmark notebook](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/benchmark_colab.ipynb),
then use `preset = "grid"`. From the repository root the command is:

```bash
uv run warptracer benchmark grid
```

It writes `outputs/grid.json` and performs three checks:

| Check | What it measures |
| --- | --- |
| Identical-pose scan comparison | Range errors, hit/miss differences and march-budget counts |
| `scan` workload | Repeated scans at identical fixed poses, without physics or controller work |
| `navigation` workload | Complete physics, sensing and disparity-control throughput for each backend |

Timing uses 1, 64 and 256 independent cars, three trials, 100 seconds of the
60 Hz schedule per trial, and the usual two-real-second warmup. Fixed scan poses
are spaced around the oval; both backends see exactly the same poses. Scan trials
do not move cars or simulate physics; their physics-substep rate is zero and
their simulation-speed metric is null. Their aggregate transition rate is the
number of complete per-car scans per second, not the number of individual rays.

The printed **GRID/MESH** ratio is mesh median time divided by grid median time:
above 1 means grid is faster. Navigation trajectories can differ because scans
differ, so use the fixed-pose workload to isolate sensor performance. Every
execution variant is separately checked against unfused/eager execution using
the **same** sensor backend; those parity checks do not imply mesh/grid agreement.

## How it works

Preprocessing rasterizes road and barrier masks once, then SciPy's Euclidean
distance transform (EDT) computes signed XY clearance. Positive values are
outside a mask; negative values are inside. The Warp kernel samples the packed
fields bilinearly and jumps along the ray according to clearance. It visits
sample positions, rather than walking every crossed cell.

| Packed channel | Meaning |
| --- | --- |
| Wall field | Signed XY distance to the barrier footprint |
| Road field | Signed XY distance to the road/base footprint; other regions are holes |
| Height | Ground elevation in meters |

EDT measures distances between lattice samples. A half-cell correction places
straight boundaries between occupied and empty samples. Bilinear interpolation
can have a gradient above one, so each distance field is scaled by its measured
maximum gradient to give conservative clearance bounds. This changes jump
lengths while preserving the zero contour. It does not make the rasterized
geometry identical to the mesh.

Ground and wall roofs also need a vertical bound. The maximum gradient of the
bilinear height field limits how quickly the ray can approach them. Combining
XY clearance with these directional height bounds allows large jumps above flat
terrain without skipping finite-height walls. A ray can hit ground, a wall side,
or a roof, start inside a wall and find its exit, or pass above a barrier.
A too-near first hit still blocks farther geometry.

Marching stops within **2 mm of the implicit surface**, or when the ray leaves
the scene/range. This is a surface-distance tolerance, not a guaranteed 2 mm
range error for a grazing ray. The budget is 2,048 iterations per ray. Rare
grazing rays in the 4,096-car navigation workload exceeded the previous
512-step limit; one reproduced ray
converged after 587 steps. Raising the ceiling preserves the tolerance and
conservative jumps; ordinary rays still stop immediately on convergence. Exhausted
rays return invalid and increment a separate counter; the comparison and timed
benchmark reject runs with exhaustion rather than accepting incomplete scans as
ordinary misses. `march_diagnostics` records tolerance, budget, observed maximum
iterations and exhaustion counts. Counts are kept on the device and copied only
for explicit inspection after timing. Warmup/reset scans are excluded from trial
counts; demo counts include its initial scan.

Optional sampling caches retain the four grid values for one ray until it leaves
its current bilinear patch. Fresh interpolation preserves the same implicit
surfaces and convergence tolerance. Default scans remain uncached pending GPU
measurements; `benchmark scale` tests both variants alongside the mesh baseline.

Device scan buffers and graph capture are reused. EDT runs once during setup,
outside benchmark timing; rays, vehicle poses and scans stay on the device.
The unchanged `racesense3d` package supplies the mesh reference.

The approach follows the clearance-and-height marching idea in
[warporacer3d](https://github.com/AlistairKeiller/warporacer3d/tree/0d8e059353d17218a69f949a558b14d84d1b305c/racer).
That revision uses Mojo/MAX; this implementation uses Warp and additionally
preserves finite barrier roofs and reports iteration-budget failures.

## Accuracy and current limits

The default CPU comparison uses 432 sensor poses: 48 angles, three lane offsets,
and road-aligned / +15° / -15° pitch. With 2.5 cm samples and 108 beams:

| Metric | Observed |
| --- | ---: |
| Rays | 46,656 |
| Common valid hits | 44,525 |
| Median absolute range error | 5.54 mm |
| 95th percentile error | 15.4 mm |
| 99th percentile error | 27.7 mm |
| Common hits within 5 cm | 99.77% |
| Hit/miss differences | 29 (0.062%) |
| Common-hit errors over 1 m | 12 |
| Maximum common-hit error | 6.86 m |
| Maximum march iterations observed | 132 of 512 |
| Iteration-budget exhaustion events | 0 |

Percentiles include only shared hits. The report separately retains validity
mismatches and the worst rays. EDT zero contours round raster corners, and small
boundary shifts can change a grazing ray's first hit by meters. These outliers
remain a limitation; ordinary range errors and exhaustion counts must be read
alongside them. Finer sampling can improve geometry but costs preprocessing and
memory, and does not guarantee identical grazing decisions.

The CPU demo completes multiple laps with zero contacts. The user's T4 EDT run
measured grid/mesh navigation throughput ratios of 1.091x / 1.075x / 1.161x at
1 / 64 / 256 cars. Fixed-scan ratios were 0.782x / 0.972x / 1.279x. Those results
precede the optional sample cache and do not predict 4,096-car or PPO throughput.
Both that run and the slower deleted traversal backend are recorded in
[VALIDATION.md](../VALIDATION.md). Keep the mesh baseline while evaluating speed
and accuracy, and use `benchmark scale` for identical moving poses and larger
batches.

Only the oval adapter is wired into `Simulation`. The underlying `DistanceField`
accepts equally shaped `[nx, ny]` NumPy arrays for height samples and binary road/
wall masks, with an empty outer mask border. Arrays use XY order; origin is the
lower sample coordinate and spacing is in meters. Obstacles share one finite
height above local ground. Overhangs, bridges and stacked surfaces need a richer
representation. PNG loading, arbitrary-map physics and slope gravity remain
future work; vehicle physics still follows the oval's analytic terrain.

For an advanced resolution experiment, override only what changes:

```bash
uv run warptracer benchmark grid --grid-cell-size .02
```

The oval adapter requires sample spacing no larger than one quarter of barrier
thickness and caps generated fields at four million samples. Python callers use
`LidarConfig(backend="grid", grid_cell_size=.025, frequency=60)` on `OvalTrack`.
Benchmark schema 5 records `lidar_algorithm = "edt-sphere-tracing"` and the
`grid_sample_cache` field distinguishes uncached/cached timing results.
