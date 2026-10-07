# Height-field LiDAR experiment

The optional grid backend casts **3D rays without a triangle mesh or BVH**.
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
| Identical-pose scan comparison | Range errors and hit/miss differences between mesh and grid geometry |
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

## Representation

| Data | Meaning |
| --- | --- |
| Vertex heights | Road elevation in meters; bilinear inside a cell |
| Surface mask | Cells containing road or an obstacle base; other cells are holes |
| Obstacle heights | Solid interval from road elevation to road elevation plus obstacle height |
| Tile bounds | Conservative minimum/maximum elevations used to skip empty space |

The oval is sampled into 2.5 cm cells. Cell centers determine road/barrier
occupancy. Rays traverse cells in XY while retaining their full XYZ direction.
Within a candidate cell, an analytic quadratic finds terrain/top intersections;
occupancy transitions expose barrier sides. Adjacent occupied cells do not
introduce internal walls. Rays can hit the road, hit a barrier roof, miss, or
pass above a barrier. A too-near first hit blocks farther surfaces, matching
the mesh sensor's range-filter behavior. Device scan buffers and graph capture
are reused; no per-scan host readback is introduced.

Eight-by-eight tiles bound all active surfaces in their cells. If a ray interval
lies entirely above, below, or outside those surfaces, it skips the tile. Tests
compare this acceleration against traversal of every cell.

Only the oval adapter is wired into `Simulation` for now. The underlying
`HeightField` accepts NumPy arrays, with shapes `[nx+1, ny+1]` for vertex heights
and `[nx, ny]` for masks/obstacle heights. Origin is the lower XY grid corner;
cell size is in meters. This is the representation for a future PNG-map loader;
image loading, scale/origin conversion and driving on arbitrary maps are not
implemented in this experiment. Physics still uses the oval's existing analytic
height and barrier-footprint checks. Overhangs, bridges and stacked surfaces
cannot be represented by this single ground layer plus solid obstacle columns.

## Accuracy and current limits

The default CPU comparison uses 432 sensor poses: 48 angles, three lane offsets,
and road-aligned / +15° / -15° pitch. Across 46,656 rays:

| Metric | Observed |
| --- | ---: |
| Common valid hits | 44,524 |
| Median absolute range error | 5.15 mm |
| 95th percentile error | 17.1 mm |
| 99th percentile error | 31.0 mm |
| Common hits within 5 cm | 99.64% |
| Hit/miss differences | 31 (0.066%) |
| Common-hit errors over 1 m | 16 |
| Maximum common-hit error | 6.82 m |

Percentiles include only rays that hit in both representations. Hit/miss
differences and worst-hit examples are reported separately, so misses are not
hidden in an average. Rasterized boundaries can catch a grazing ray that misses
the mesh, or vice versa, changing the first hit by meters despite a small
boundary displacement. Grid point clouds can therefore differ from the rendered
mesh, especially near corners and tangencies. A finer grid can reduce ordinary
errors but increases memory/traversal cost and does not guarantee identical
grazing decisions.

The CPU grid demo completes multiple laps with zero barrier contacts. GPU
execution and any speedup must still be checked on Colab; a height field is not
automatically faster than an accelerated mesh. Keep the mesh reference while
evaluating accuracy and speed together.

For an advanced resolution experiment, override only what changes:

```bash
uv run warptracer benchmark grid --grid-cell-size .02
```

The oval adapter requires cells no larger than one quarter of barrier thickness
and caps generated fields at four million cells. Python callers select it with
`LidarConfig(backend="grid", grid_cell_size=.025, frequency=60)` on `OvalTrack`.
