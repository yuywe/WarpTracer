# WarpTracer

A minimal racing simulator with a box car and vehicle-mounted LiDAR. Choose a
flat room or an oval road with gentle elevation changes. The car can accelerate,
brake, and steer. A reactive
disparity controller navigates from LiDAR; Viser exports an interactive replay.

Work is on the `performance-prototype` branch. The default lean physics model
follows the road surface; the room remains flat. Newton is available for the
original room experiments.
Batched cars run independently and cannot see or collide with each other.

## Get started

Install Git and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone --branch performance-prototype https://github.com/yuywe/WarpTracer.git
cd WarpTracer
uv run warptracer demo
```

For an existing checkout, switch to `performance-prototype` and run
`git pull --ff-only` before the demo. Run commands from the repository root;
uv installs dependencies and uses one environment there. The first simulation
run compiles kernels.

The demo runs **30 simulated seconds of LiDAR navigation**, using lean physics,
108 beams at 60 Hz, graph execution and fused substeps. It selects CUDA when
available and otherwise uses CPU. Replay dependencies are included automatically.

Open **`outputs/disparity.html`** in your browser to orbit, pause and scrub.
The matching `.npz` saves poses, controls and scans; `.json` saves the settings.
The replay shows recorded motion; moving its camera does not rerun physics.

## Common commands

| What you want | Command | Output |
| --- | --- | --- |
| LiDAR navigation replay | `uv run warptracer demo` | `outputs/disparity.html` |
| Elevated oval track replay | `uv run warptracer demo oval` | `outputs/oval.html` |
| Acceleration and braking check | `uv run warptracer demo accelerate-brake` | `outputs/accelerate-brake.html` |
| Circle driving check | `uv run warptracer demo circle` | `outputs/circle.html` |
| Quick performance check | `uv run warptracer benchmark` | `outputs/quick.json` |
| LiDAR vs. disparity navigation | `uv run warptracer benchmark navigation` | `outputs/navigation.json` |
| Navigation on the elevated oval | `uv run warptracer benchmark oval` | `outputs/oval-benchmark.json` |
| Independent car scaling | `uv run warptracer benchmark batches` | `outputs/batches.json` |
| Fused vs. unfused physics | `uv run warptracer benchmark fusion` | `outputs/fusion.json` |

The `oval`, `navigation`, `batches` and `fusion` presets require CUDA by default.
They fail immediately if a GPU is unavailable. The quick benchmark also works
on CPU. See [benchmark presets and result units](docs/benchmarking.md).

The oval demo runs for 120 simulated seconds on a 16 × 10 m loop with a nominal
2 m lane and 0.4 m elevation range. The chassis follows road height and tilt;
LiDAR and replay use the same 3D track mesh. Open `outputs/oval.html` and check
the printed lap/contact counts. See [tracks and terrain following](docs/tracks.md).

Only add an option when you need to change something:

```bash
uv run warptracer demo --seconds 60
```

Use `uv run warptracer demo --help` or `uv run warptracer benchmark --help`
for advanced settings. Existing `warptracer-demo` and Python module commands
remain available.

## Colab

Select **Runtime → Change runtime type → GPU**, then open either notebook:

- [Driving replay](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/rigidbody_colab.ipynb): run setup, simulate, then display. It defaults to `scenario = "oval"`.
- [Benchmarks](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/benchmark_colab.ipynb): run setup, choose a `preset`, then read the results. It defaults to `preset = "oval"`.

Both reuse the same checkout and root uv environment. Output streams into the
cell. No command construction or physics settings need editing for normal runs.

## Details and development

- [Navigation behavior and limits](docs/navigation.md)
- [Tracks and terrain following](docs/tracks.md)
- [Benchmark presets, timing, and batch API](docs/benchmarking.md)
- [Vehicle, LiDAR, controls, and saved arrays](docs/reference.md)
- [Validation results and model limits](VALIDATION.md)

Run tests with `uv run --extra dev pytest -q`.
