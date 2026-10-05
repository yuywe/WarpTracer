# LiDAR navigation

Run from the repository root:

```bash
uv run --locked --extra viz warptracer-demo --scenario disparity --device cuda:0 --seconds 30 --lidar-hz 60
```

Use `--device cpu` without CUDA. Open `outputs/disparity.html` for an interactive
replay. The NPZ retains poses, applied controls and scans; the JSON records
controller settings and `wall_contact_substeps`. This counter counts every
physics substep touching a wall, including contacts inside a fused transition.
It is not a count of distinct crashes. Newton currently reports no such counter.

## Controller

The implementation follows the range-disparity extension described by
[Nathan Otterness](https://nathanotterness.com/2019/04/the-disparity-extender-algorithm-and.html).
It finds adjacent range jumps and extends the nearer distance into the farther
side by half the chassis width plus a margin. Overlaps keep the nearest distance.
Original scans are preserved. The controller selects a long clear direction
within 80 degrees of forward, with a small preference for straighter targets.

Speed decreases with steering and forward clearance. Braking uses the existing
speed controller and tire grip limits; steering keeps its existing rate limit.
The braking allowance also checks the selected target's filtered distance,
accounting for the sensor position and half the chassis length.
Invalid or near-clipped readings are conservatively treated as blocked. The
sensor's validity mask does not distinguish a miss from a near-clipped obstacle.

Smooth room corners may have no range disparity. An additional corner rule starts
a turn before clearance drops below a turning-diameter allowance, including
steering response time and margins. It chooses the more open side and holds the
turn until forward clearance recovers. Corner targets use a real beam inside the
search sector and its filtered clearance, including the same braking and stop
checks. Turn hysteresis uses valid sensor ranges up to the sensor's far limit,
independently of the shorter range cap used to score and extend targets. This
allows a turn to release even when the scoring cap is below the release threshold.
This avoids the prototype's tendency to
drive into a corner and stop. A side-clearance stop remains as a fallback.

These are prototype heuristics for the flat enclosure, not a collision guarantee
or a racing policy. There is no reversing/recovery planner: a blocked or narrow
situation can leave the car stopped. No map, localization, reward or PPO is used.

We also reviewed the controller in
[uci-f1tenth/race_stack](https://github.com/uci-f1tenth/race_stack/blob/main/disparity_extender/disparity_extender/disparity_extender.py).
Despite its name, that version selects the center of a sliding window with the
greatest minimum clearance; it does not extend adjacent disparities. It uses
clearance at the selected target to limit speed. Our controller retains actual
disparity extension, physical steering angles and braking limits. Its 300-beam
window and normalized steering settings are not directly usable with our
108-beam scan. ROS scan-dropout handling is unnecessary for this synchronous
simulation, where each transition produces the next scan.

## Execution and configuration

Control consumes the current scan, advances physics, then produces the next scan.
Reset produces a fresh initial scan and clears the corner-turn state. Control
runs once per scan, with no scan copies to the CPU. Each batched car has separate
commands and controller state and cannot see the other cars.

The replay defaults to 30 Hz sensing/control. Set `--lidar-hz 60` for the benchmark
rate. At 40 Hz, use `--lidar-hz 40 --record-fps 20`. Duration and recording intervals
must contain whole transitions, each with an even number of physics substeps.

```python
from warptracer.disparity import DisparityConfig
from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation

sim = Simulation(scenario="drive", engine="lean", device="cuda:0", num_envs=256,
                 lidar=LidarConfig(beams=108, frequency=60))
runner = TransitionRunner(sim, backend="graph", integrator="fused", substeps=4,
                          controller="disparity",
                          disparity_config=DisparityConfig(max_speed=1.5))
runner.advance()
# runner.navigator.filtered and .goals are borrowed device arrays.
# .goals contains [target angle radians, best filtered distance meters];
# both values refer to the selected beam, including corner overrides.
runner.reset()
```

The controller requires an unrotated, horizontal LiDAR mount, at least three
beams, forward coverage, at least one beam inside the configured search sector,
and a field of view from 180 up to (excluding) 360
degrees. Defaults are tested with 108 beams. The extension pass currently checks
all disparities per output beam (quadratic in beam count); dense scans increase
controller work as well as raycasting work.

Calling `DisparityController.update()` directly also invalidates the scalar
command cache, allowing subsequent manual setters to replace device commands.

Use `--cases navigation` in `warptracer.benchmark` to measure physics, LiDAR and
control together. Existing `physics`, `lidar` and `recording` cases keep their
scripted circle controller so previous measurements remain comparable.

On a Colab GPU runtime, compare navigation against the LiDAR baseline:

```bash
uv run --locked python -u -m warptracer.benchmark --device cuda:0 --physics lean --backend graph --integrator fused --envs 1 64 256 --cases lidar navigation --seconds 100 --trials 3 --output outputs/navigation-benchmark.json
```

This comparison measures both the added controller work and its changed vehicle
trajectory. It is not an isolated timing of the disparity kernels.
