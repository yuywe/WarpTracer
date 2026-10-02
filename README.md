# WarpTracer

A minimal racing simulator: **one box car, one flat floor, four walls, and
vehicle-mounted LiDAR**. The car can accelerate, brake, and steer. Viser exports
an interactive replay that you can orbit, pause, and scrub.

This branch is `performance-prototype`. Driving uses a lightweight Warp model;
Newton is also available. The car currently follows scripted commands.
Autonomous navigation and parallel environments are the next milestones.

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
| `accelerate-brake` | Accelerates straight, then stops |
| `circle` | Drives with constant left steering |
| `s-turn` | Alternates left and right steering, then stops |

Each lasts ten simulated seconds. Add options to the same command:

| Option | Effect |
| --- | --- |
| `--seconds 20` | Run for 20 simulated seconds |
| `--device cpu` | Force CPU execution |
| `--headless` | Skip HTML replay and intermediate recording |
| `--no-lidar` | Disable LiDAR |
| `--physics newton` | Use the original Newton physics |

The default lean model includes acceleration, braking, steering, sideways slip,
grip limits, and stopping at walls. Height is fixed; suspension, roll/pitch, and
uneven terrain are not modeled. The `drop` and `wall-impact` scenarios always use
Newton.

## Run in Colab

Open the [driving notebook](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/rigidbody_colab.ipynb)
and run its three code cells: **setup → simulate → display**.
Change the scenario in the second cell to try another drive.
Setup reuses the same clone and repository-root uv environment.

## Measure performance

From the repository root, run:

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --backend both
```

Use `--device cpu` without an NVIDIA GPU, or open the
[benchmark notebook in Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/benchmark_colab.ipynb).

The command compares ordinary execution (`eager`) with captured graph replay
(`graph`) for physics alone, physics with LiDAR, and physics with LiDAR plus
host recording. Results are written to **`outputs/benchmark.json`**.

**One transition = four physics substeps.** The benchmark runs one car at
240 physics substeps per simulated second, with 1080 LiDAR rays at 60 Hz.
The driving replay uses 30 Hz LiDAR. Compare matching workloads and units.

In a user-supplied RTX 4060 Laptop GPU run, lean graph execution with LiDAR
reached **29,360 transitions/s** (117,442 physics substeps/s), a **6.7×**
improvement over eager execution. All benchmark parity checks passed.
See [validation results](VALIDATION.md) for timings and measurement limits.

## Further details

- [Vehicle, controls, LiDAR, and saved arrays](docs/reference.md)
- [Benchmark options and graph execution](docs/benchmarking.md)
- [Tests, measured results, and model limits](VALIDATION.md)

Run the test suite with:

```bash
uv run --locked --extra viz --extra dev pytest -q
```
