# WarpTracer

A minimal racing simulator: **one box car, one flat floor, four walls, and
vehicle-mounted LiDAR**. The car can accelerate, brake, and steer. Viser exports
an interactive replay that you can orbit, pause, and scrub.

This branch is `performance-prototype`. Driving uses a lightweight Warp model;
Newton is also available. Drive with scripted commands or a reactive LiDAR
disparity extender. Independent cars can run in batches on the same static track.

## Run locally

You need Git and uv. For a new checkout:

```bash
git clone --branch performance-prototype https://github.com/yuywe/WarpTracer.git
cd WarpTracer
```

If you already cloned the project, run these inside your existing repository:

```bash
git fetch origin
git switch performance-prototype
git pull --ff-only origin performance-prototype
```

Then install dependencies and run the car:

```bash
uv sync --locked --extra viz
uv run --locked --extra viz warptracer-demo --scenario accelerate-brake
```

Run all commands from the repository root so uv uses one environment.
CUDA is selected when available; CPU also works. The first run compiles kernels.

Open **`outputs/accelerate-brake.html`** in your browser. The replay contains
recorded motion and LiDAR; moving the camera does not rerun the simulation.
The matching `.npz` stores arrays and `.json` stores configuration and timing.

## Choose a driving scenario

Replace `accelerate-brake` in the command above:

| Scenario | What the car does |
| --- | --- |
| `disparity` | Chooses steering and speed from LiDAR; turns away from walls |
| `accelerate-brake` | Accelerates straight, then stops |
| `circle` | Drives with constant left steering |
| `s-turn` | Alternates left and right steering, then stops |

Each lasts ten simulated seconds. Add options to the same command:

| Option | Effect |
| --- | --- |
| `--seconds 20` | Run for 20 simulated seconds |
| `--device cpu` | Force CPU execution |
| `--headless` | Skip HTML replay and intermediate recording |
| `--lidar-beams 1080` | Restore dense scans (default: 108 rays) |
| `--no-lidar` | Disable LiDAR |
| `--physics newton` | Use the original Newton physics |

The default lean model includes acceleration, braking, steering, sideways slip,
grip limits, and stopping at walls. Height is fixed; suspension, roll/pitch, and
uneven terrain are not modeled. The `drop` and `wall-impact` scenarios always use
Newton.

Try autonomous navigation and open **`outputs/disparity.html`**:

```bash
uv run --locked --extra viz warptracer-demo --scenario disparity --seconds 30 --lidar-hz 60
```

This uses a 1.5 m/s target-speed limit, slows for turns and obstacles, and reports
wall-contact substeps. It needs LiDAR; `--max-speed` adjusts its speed limit.
See [navigation behavior and limits](docs/navigation.md). The scene still has
only the box chassis, floor, four walls, and LiDAR display.

## Run in Colab

Open the [driving notebook](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/rigidbody_colab.ipynb)
and run its three code cells: **setup → simulate → display**.
Change the scenario in the second cell to try another drive.
Setup reuses the same clone and repository-root uv environment.
The notebook defaults to disparity navigation and `cuda:0`; select a GPU runtime
or explicitly change `device` to `"cpu"`. Output is streamed into the cell.

## Measure performance

From the repository root, run:

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --backend both
```

Use `--device cpu` without an NVIDIA GPU, or open the
[benchmark notebook in Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/benchmark_colab.ipynb).

The command compares ordinary execution (`eager`) with captured graph replay
(`graph`). Lean benchmarks default to fused physics substeps, and warm up for
at least two real seconds before each trial. Results go to **`outputs/benchmark.json`**.

**One transition = four physics substeps.** The default benchmark runs one car at
240 physics substeps per simulated second, with 108 LiDAR rays at 60 Hz.
The driving replay uses 30 Hz LiDAR. Compare matching workloads and units.

Compare fusion first, then sweep independent cars:

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --backend graph --integrator both --envs 1 --cases physics lidar --seconds 1000 --output outputs/fusion.json
uv run --locked python -m warptracer.benchmark --device cuda:0 --backend graph --integrator fused --envs 1 64 256 1024 --cases physics lidar --seconds 100 --output outputs/batches.json
```

**Aggregate transitions/s** counts all cars; **batch transitions/s** counts
advances of the whole batch. Each car has its own state and scan, sharing the
same static track geometry. See the [benchmark guide](docs/benchmarking.md)
for 40 Hz sensing, timing interpretation, and the Python API.

Include the LiDAR controller in a headless benchmark:

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --backend graph --integrator fused --envs 1 64 256 --cases navigation --seconds 100 --output outputs/navigation.json
```

## Further details

- [Vehicle, controls, LiDAR, and saved arrays](docs/reference.md)
- [Benchmark options and graph execution](docs/benchmarking.md)
- [Tests, measured results, and model limits](VALIDATION.md)

Run the test suite with:

```bash
uv run --locked --extra viz --extra dev pytest -q
```
