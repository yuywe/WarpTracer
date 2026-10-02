# Benchmarking and graph execution

[Quick start](../README.md) · [Validation results](../VALIDATION.md)

## Compare fusion

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --backend graph --integrator both --envs 1 --cases physics lidar --seconds 1000 --output outputs/fusion.json
```

Unfused integration launches a kernel for each physics substep. Fused integration
runs the same substeps sequentially inside one kernel per transition, keeping
each car's intermediate state local. Both retain the 1/240-second physics dt,
steering rate limiting, speed control, grip, and collision checks on every substep.
Only final state is written to the arrays. Commands stay fixed for the transition.

Graph execution captures the sequence once and replays it. Fusion reduces the
number of operations inside that graph; these are separate optimizations.
`--backend both` also measures eager Python submission.

Lean benchmarks default to `--integrator fused`. Pass `--integrator unfused`
to compare with earlier reports. Newton supports only unfused, single-car runs.
Driving replay demos and `Simulation.step()` still use single-step integration.

## Sweep independent cars

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --backend graph --integrator fused --envs 1 64 256 1024 --cases physics lidar --seconds 100 --output outputs/batches.json
```

One physics thread advances each car; ray casting spans cars and beams. The
batch shares one immutable wall/floor mesh. Cars live in independent copies of
that enclosure: they neither collide with nor see each other. Vehicle and track
parameters are shared; state, controls, collisions, and scans are separate.
The benchmark gives all cars the same scripted circle commands. Tests also use
different per-car commands to check isolation.

`--seconds` is simulated duration **per car**, not total duration across the batch.
There are no policy inference, rewards, auto-resets, or RL training in this test.

## Units and defaults

| Setting / field | Meaning |
| --- | --- |
| Physics | 240 substeps per simulated second |
| `--substeps 4` | Four sequential physics updates per transition |
| LiDAR | 108 rays, one scan per transition; 60 Hz by default |
| `--lidar-beams 1080` | Restore the earlier dense scan workload |
| `--envs 1` | One car by default; accepts a list for sweeps |
| `--trials 5` | Five trials per configuration |
| Aggregate environment transitions/s | Cars × batch transitions ÷ wall time |
| Batch transitions/s | Runner advances ÷ wall time, also transitions/s per car |
| Physics substeps/s | Cars × transitions × substeps ÷ wall time |
| Simulated seconds/s per environment | Simulation speed relative to real time for each car |

To use **40 Hz LiDAR**, add `--substeps 6` to either headless command above.
For the recording case at 40 Hz, also use `--record-hz 20` or `40`.
The recording rate must divide the transition rate; headless cases ignore it.

For a matched ray-count comparison, repeat an identical command with
`--lidar-beams 108` and `--lidar-beams 1080`, using different output files.
At 108 rays the angular spacing is about 2.52°. Spatial resolution and navigation
quality still need evaluation; ten times fewer rays does not imply ten times
higher overall throughput.

## Warmup and measurement

Before **each trial**, the runner executes until both minima are satisfied:
`--warmup-wall-seconds 2` real seconds and `--warmup-seconds 2` simulated seconds.
Work is submitted in chunks of 32 transitions and synchronized between chunks.
The report saves actual warmup duration and count. Reset and synchronization
follow warmup, outside the timed region. Longer warmup may help exercise the GPU,
but does not guarantee stable clocks or explain earlier timing variation.

Every fused or graph variant is checked against unfused eager execution before
timing: all cars' poses, velocities, motion states, controls, collision flags,
scan poses, ranges, masks, and device step counts must agree within tolerance.
The reference uses the same batch size; separate tests compare batch entries
against independent single-car simulations.

Measured time includes stepping, sensing, and optional host recording. It excludes
construction, compilation, warmup, reset, parity validation, HTML export, and
disk writes. Cases are `physics`, `lidar`, and `recording`; omit `--cases` to
run all three. Recording copies and retains all cars' states/scans at the
recording rate, so use headless cases for large, long-running batch sweeps.

Reports use schema version 2. Each result carries environment count, integrator,
backend, case, individual trials, median, min/max, and max/min ratio. A ratio above
1.2 produces a `timing_variable` flag; it is a diagnostic, not a statistical test.
The `validation` list replaces the old `graph_validation` mapping.
The older `environment_transitions_per_second` field remains as an alias for
aggregate throughput. Final states are checked for every car, but only car zero
is stored in each trial's `final_pose` and `final_velocity`.

CUDA event intervals include stream idle gaps and are not summed kernel durations.
`--profile` runs a separate eager Python call profile for the smallest requested
batch; it includes that extra run's warmup and does not profile CUDA kernels.

## Python API

```python
import numpy as np
from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation

sim = Simulation(scenario="drive", engine="lean", device="cuda:0", num_envs=64,
                 lidar=LidarConfig(frequency=40))
runner = TransitionRunner(sim, backend="graph", integrator="fused", substeps=6)

# Columns: throttle, brake, steering radians, target speed m/s.
# A target of -1 selects direct throttle/brake; >= 0 selects speed control.
commands = np.zeros((64, 4), dtype=np.float32)
commands[:, 2] = np.linspace(-.2, .2, 64)
commands[:, 3] = 1.0
sim.set_commands(commands)
for _ in range(40):
    runner.advance()
poses, velocities = sim.snapshot()  # explicit host copy: (64, 7), (64, 6)
sensor_poses, ranges, valid = sim.lidar.snapshot()  # (64, 7), (64, 108), (64, 108)
runner.reset()
```

Scalar setters broadcast to all cars. GPU controllers can write the stable
`sim.device_commands` buffer directly on the same stream; call
`sim.invalidate_command_cache()` before returning to scalar setters.
LiDAR `result.values` and `result.valid` remain device arrays shaped
`(num_envs, beams)`. Buffers are reused each transition.

The runner owns stepping/reset; do not interleave `sim.step()` or `sim.reset()`.
Reset currently resets the whole batch. Collision flags describe the final
physics substep, not a latched episode termination. `Simulation.run()` and Viser
exports support one car; use the runner for batches.

Use `--device cpu` without CUDA. Warp's CPU API capture can test graph behavior,
but CPU timings are not GPU throughput measurements. Automatic backend selection
uses both eager/graph on CUDA and eager on CPU.
