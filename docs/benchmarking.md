# Benchmarks

[Quick start](../README.md) · [Validation results](../VALIDATION.md)

## Choose a preset

```bash
uv run warptracer benchmark navigation
```

Presets set the workload, car counts, execution mode, duration and output path.
All use lean physics, 108 LiDAR beams, 240 Hz physics, 60 Hz transitions and three
trials. Every trial warms up for at least two real and two simulated seconds.

| Preset | Comparison | Cars | Simulated seconds per trial | Output |
| --- | --- | --- | ---: | --- |
| `quick` (default) | Eager vs. graph; physics, LiDAR, recording | 1 | 10 | `outputs/quick.json` |
| `navigation` | LiDAR baseline vs. live disparity control | 1, 64, 256 | 100 | `outputs/navigation.json` |
| `oval` | Physics, LiDAR and disparity on the elevated loop | 1, 64, 256 | 100 | `outputs/oval-benchmark.json` |
| `scale` | Mesh vs. uncached/cached EDT; moving scans and distributed oval navigation | 256, 1024, 4096 | 20 | `outputs/scale.json` |
| `grid` | Mesh vs. EDT marching: fixed scans and oval navigation | 1, 64, 256 | 100 | `outputs/grid.json` |
| `batches` | Physics and LiDAR as car count increases | 1, 64, 256, 1024 | 100 | `outputs/batches.json` |
| `fusion` | Fused vs. unfused physics; physics and LiDAR | 1 | 1000 | `outputs/fusion.json` |

`quick` chooses CUDA when available, otherwise CPU, and uses fused physics.
The other presets require CUDA and use graph replay; `navigation` and `batches`
use fused physics, as do `oval`, `grid` and `scale`. These presets use the elevated
track; the existing presets keep the room workload for comparable measurements.
Select a GPU runtime in Colab. For a small CPU check use
`uv run warptracer benchmark`; CPU timings do not measure GPU performance.

Duration is **per car**. Batched cars have independent states and scans in copies
of the same static enclosure. They do not interact. Physics/LiDAR/recording cases
use scripted circle commands. Navigation consumes scans to choose controls, so
its comparison includes both controller work and the changed trajectory.
These benchmarks contain no learning, policy inference, rewards or auto-resets.

`grid` also measures **fixed-pose scans without physics or controller work**.
Both sensor backends see identical car poses spaced around the track. For that
case the aggregate transition rate is complete per-car scans/s, physics rate
is zero, and simulation-speed metrics are null. The preset measures mesh/grid
scan errors before timing and prints their median-time speed ratio. See the
[grid experiment](grid-lidar.md) for accuracy limits and how to read the report.

## Large batches for future PPO rollouts

```bash
uv run warptracer benchmark scale
```

This compares mesh, ordinary EDT and cached EDT at **256, 1,024 and 4,096 cars**.
It uses two workloads, 20 seconds of the 60 Hz schedule and three trials:

| Workload | Purpose |
| --- | --- |
| `moving_scan` | Every backend follows the same prescribed 3D centerline poses, isolating sensing cost while poses change |
| `navigation` | Full lean physics, LiDAR and disparity control; starts are distributed around the oval |

Moving scans replay one 4,096-pose tilted loop from device memory, with a phase
offset per car. Each transition advances two pose samples. One shared 112 KiB
path is uploaded per runner; memory does not grow as frames times cars. Pose
replay and mount composition are included in timing; preprocessing is excluded.
No physics or policy is run in this workload, so its physics rate is zero and
its simulated-speed metric is null. This prescribed path is not a recorded
physical trajectory. Navigation remains an end-to-end comparison and its paths
can differ between sensing backends.

EDT caching retains a ray's four grid samples while it remains in one bilinear
patch. Interpolation and convergence checks still run at every step; the cache
ends with that ray. Separate specialized kernels let uncached scans avoid the
cache's conditional sampling path. `grid_sample_cache` identifies each variant;
**CACHED/UNCACHED** above one means caching was faster. Defaults remain uncached
until native GPU measurements support a change. `scale` automatically tests both.
The scan geometry, tolerance and iteration budget are unchanged.

The lean, disparity and EDT modules skip unused environment backward-code
generation. These modules serve model-free rollouts; this does not disable
backpropagation through a future PPO policy network. The original Newton and
mesh sensor packages are unchanged.

At 4,096 cars, one 108-beam scan contains **442,368 rays**. A faster sensor can
reduce rollout collection time, but total PPO speed also includes inference,
learning, rewards/resets and rollout-buffer work. These benchmarks implement
none of those PPO components. As an illustration, a 16% rollout speedup gives
about a 7% total speedup if collection originally took half the training time.
That is arithmetic under an assumed time split, not a prediction for 4,096 cars.

## Read the results

Each run prints progress and writes its JSON report. Repeating a preset overwrites
its previous report; copy it or choose `--output` to keep multiple runs.

| Metric | Meaning |
| --- | --- |
| Batch transitions/s | Advances of the entire batch per wall-clock second; also the rate per car |
| Aggregate transitions/s | Batch transitions/s × number of cars |
| Aggregate physics substeps/s | Aggregate transitions/s × substeps per transition |
| Simulated seconds/s per car | Simulation speed relative to real time |
| `timing_variable` | Slowest/fastest trial time exceeds 1.2; does not identify a cause |

One transition normally contains **four 1/240-second physics substeps and one
LiDAR scan**. Thus control and sensing run at 60 Hz. Count transitions separately
from physics substeps when comparing simulators.

Elapsed time includes Python submission, GPU waiting, physics, sensing and any
recording copies. It excludes setup, compilation, validation, warmup, reset and
disk writes. It is end-to-end throughput, not pure kernel throughput.

## Change a setting only when needed

Options override preset settings. For example, keep the navigation workload but
run one car:

```bash
uv run warptracer benchmark navigation --envs 1
```

| Option | Use |
| --- | --- |
| `--seconds 1000` | Change simulated duration per trial |
| `--envs 1 64` | Choose independent car counts |
| `--trials 5` | Change repeats per configuration |
| `--device cpu` | Explicit CPU execution; use small batches and short durations |
| `--lidar-beams 1080` | Dense scan comparison |
| `--substeps 6` | 40 Hz control and sensing; recording also needs `--record-hz 20` or `40` |
| `--output outputs/my-run.json` | Save to a different path |
| `--track oval` | Use the elevated loop in a custom lean workload |
| `--lidar-backend grid` | Optional EDT ray marching on the oval; `both` compares against mesh |
| `--grid-cell-size .02` | Change grid resolution in meters |

Use `uv run warptracer benchmark --help` for all options. Low-level module
commands remain supported for custom experiments, including Newton (one car,
unfused integration). Their original defaults are preserved when no preset is
given.

## Graphs, fusion and validation

Eager execution submits operations from Python each transition. Graph execution
captures the operation sequence once and replays it. Unfused physics launches a
kernel per substep; fused physics advances those same substeps inside one kernel.
Graph replay and fusion work together. Both preserve the physics timestep,
steering rate limits, tire grip and collision checks at every substep.

Every fused or graph variant is compared against unfused eager execution before
timing. Validation checks all cars' poses, velocities, controls, motion states,
contact counts, scans and clocks; navigation also checks filtered scans and goals.
Separate tests compare batch entries against independent single-car runs,
including selected entries of a 4,096-car moving-scan batch.
For the grid and scale experiments, parity uses the same LiDAR backend on both sides.
Mesh/grid geometry differences are measured separately at identical poses;
they are not asserted to be exact parity.

Warmup submits full-trial chunks and checks both duration minima between chunks,
so it can overshoot by one chunk. The report records the actual warmup. CUDA event
intervals include stream idle gaps; they are not summed kernel durations.
`--profile` adds a separate eager Python profile, not a CUDA kernel profile.

## Python batch API

```python
import numpy as np
from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation

sim = Simulation(scenario="drive", engine="lean", device="cuda:0", num_envs=64,
                 lidar=LidarConfig(frequency=40))
runner = TransitionRunner(sim, backend="graph", integrator="fused", substeps=6)
commands = np.zeros((64, 4), dtype=np.float32)
commands[:, 2] = np.linspace(-.2, .2, 64)
commands[:, 3] = 1.0
sim.set_commands(commands)
for _ in range(40):
    runner.advance()
poses, velocities = sim.snapshot()  # Host copy: (64, 7), (64, 6)
sensor_poses, ranges, valid = sim.lidar.snapshot()  # (64, 7), (64, 108), (64, 108)
runner.reset()
```

Command columns are throttle, brake, steering radians and target speed m/s.
Target speed `-1` selects direct throttle/brake; `>= 0` selects speed control.
Scalar setters broadcast to all cars. Device controllers can write the stable
command buffer on the simulation stream; invalidate the command cache before
returning to scalar setters. LiDAR device buffers are reused each transition.

The runner owns stepping/reset; do not interleave `sim.step()` or `sim.reset()`.
Reset affects the whole batch. Collision flags describe the final substep;
contact counters accumulate all contacting substeps. Recorded demos support one
car. Use the runner for batches.
