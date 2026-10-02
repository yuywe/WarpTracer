# WarpTracer — lean vehicle simulation

A box car that accelerates, brakes, and steers on a flat surface with four walls,
with the existing 1080-beam LiDAR mounted on the chassis. The
`performance-prototype` branch adds a lightweight Warp dynamic bicycle backend,
device-resident controls, graph replay, and a repeatable single-car benchmark.
Driving demos default to the lean backend. Newton remains available for comparison
and the free-body drop/impact checks. LiDAR observes the scripted driving;
autonomous navigation is a later milestone.

[Driving replay in Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/rigidbody_colab.ipynb) ·
[Benchmark in Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/benchmark_colab.ipynb)

## Run in Colab

Run the three cells: setup, simulate, display. Setup reuses the earlier
`/content/WarpTracer-rigidbody` clone, switches to `performance-prototype`, and updates
it with a fast-forward pull. Colab's preinstalled `uv` uses **one environment at
that repository root** and Colab's Python interpreter. Local edits that conflict
with switching/pulling must be resolved; setup does not discard them.

Choose `accelerate-brake`, `circle`, or `s-turn` in the second cell. Each driving
recording lasts ten seconds by default. Rerun the last two cells after changing
scenarios. CPU works; CUDA is selected automatically when available. The first
execution compiles Warp kernels.

The replay has orbit/zoom controls and a timeline. It contains recorded motion;
camera movement does not rerun physics or control the car. Yellow points are
LiDAR returns and green lines are sampled laser rays. No sensor housing mesh is
added: the ray origin shows the mounting position on the box.

## Run with uv

From the repository root:

```bash
uv sync --locked --extra viz
uv run --locked --extra viz warptracer-demo --scenario accelerate-brake
uv run --locked --extra viz warptracer-demo --scenario circle
uv run --locked --extra viz warptracer-demo --scenario s-turn
```

Open the corresponding `outputs/<scenario>.html` in a browser. Each HTML file
includes the viewer and recording, so playback needs no persistent Python server.

The earlier `drop` and `wall-impact` checks remain available, with four-second
default durations. Options include `--seconds 10`, `--device cpu|cuda:0`,
`--physics-hz 240`, `--record-fps 30`, `--output outputs`, and `--headless`.
LiDAR is enabled by default; `--no-lidar` runs the vehicle-only demo and
`--lidar-hz 30` sets scan frequency. Driving requires at least 120 physics steps/s;
both pose recording frequency and LiDAR frequency must divide physics frequency.
Use the default 240 Hz. `--physics newton` selects the original spring/contact model.
The drop and wall-impact demos always use Newton.

Each run writes pose, velocity, and applied control arrays (`.npz`) and
configuration/timing (`.json`). LiDAR arrays are included when enabled (details
below). Recorded runs also write `.html`. Headless runs still compute LiDAR at
its configured frequency, but copy only the initial and latest scan to the CPU,
along with initial/final body states. They use a `_headless` filename suffix.
Outputs are ignored by Git.

## Vehicle-mounted LiDAR

The four Python library files in `src/racesense3d/` are copied **unchanged** from
[`sensor-prototype` commit 51b2d1c](https://github.com/yuywe/WarpTracer/tree/51b2d1c8a4c4460d1b1828e5b0240ee287ba577d/racesense3d/src/racesense3d).
`warptracer/lidar.py` adapts that reusable ray caster to the vehicle and track.
Both packages use the existing root uv project; there is no nested project or
second environment. Only LiDAR is instantiated; camera code is not used.

| Setting | Default |
| --- | --- |
| Beam layout | 1080 beams in one horizontal ring, -135° to +135°, ordered right to left |
| Range | 0.021 m minimum, 30 m maximum |
| Rate | 30 Hz, independently of pose recording rate |
| Mount translation | 0.12 m forward, centered laterally, 0.025 m above the chassis top |
| Mount orientation | Aligned with the chassis; follows its yaw, pitch, and roll |
| Scanned surfaces | The same four wall boxes and floor as the physics scene |
| Host vehicle | Excluded from the ray-casting mesh |
| Replay | Every sixth return and every 36th ray; full-resolution scans are saved |

The static sensor mesh is built once from `Track.barriers()`, so wall dimensions
and positions match physics and playback. A floor patch extends beyond the walls
by the maximum sensing range. This covers floor hits while the car is inside the
enclosure. The added ray-casting mesh has 50 triangles and is never rendered.

Each scan uses the composed **world-from-body × body-from-sensor** transform.
The pose and ray-casting kernels run on the simulation device; sensing adds no
forces or bodies. Scans are instantaneous snapshots, with no rotating-scan timing,
noise, or dropout. A miss or filtered hit returns `far` with `valid=False`;
too-near surfaces still occlude anything behind them.

```python
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation

sim = Simulation(scenario="circle", engine="lean", lidar=LidarConfig(), device="cpu")
trajectory = sim.run(duration=10)
scans = trajectory.lidar
print(scans.ranges.shape)   # (301, 1080): initial scan plus 30 scans/s
points = scans.points(30)  # scan at 1 second, world XYZ; invalid returns are NaN
```

For manual stepping, `sim.lidar.result` exposes the existing `Scan` object and
its device-resident `values`/`valid` buffers. `sim.lidar.timestamp` gives the latest
scan time. Buffers are reused by the next scan; copy explicitly if retaining them.
The controller sees the latest completed scan before the next physics step.
`reset()` restores the car and immediately refreshes the time-zero scan.
Python callers can omit `lidar` to run the original sensor-free simulation.

NPZ data includes:

| Array | Shape / meaning |
| --- | --- |
| `lidar_times` | `(scans,)`, seconds from reset |
| `lidar_poses` | `(scans, 7)`, sensor world position + XYZW quaternion |
| `lidar_ranges` | `(scans, beams)`, meters |
| `lidar_valid` | `(scans, beams)`, boolean |
| `lidar_directions` | `(beams, 3)`, unit rays in sensor coordinates |

Use the LiDAR timestamps when pairing scans with vehicle poses: the rates can
differ. A run ending between scan ticks keeps the last scheduled scan rather
than inventing an extra sample. Replay holds each scan in world coordinates
until the next scan, while pose frames follow their own timeline.

## Driving model

Both backends use the same single 0.52 × 0.26 × 0.12 m, 3.2 kg box, rectangular
track, controls, LiDAR, and playback. Driving uses a 12 × 6 m clear area,
or 8 × 8 m for the circle demo. There are no wheel meshes or joints.

| Part | Lean backend | Newton baseline |
| --- | --- | --- |
| Motion | Forward/sideways velocity and yaw rate; dynamic bicycle | Six-degree-of-freedom rigid body |
| Tire forces | Two axles with slip-velocity damping and shared longitudinal/lateral grip budgets | Four tire points with friction limited by spring support |
| Steering | Front axle, ±0.418 rad limit, 1.5 rad/s rate limit | Same limits |
| Acceleration/braking | Grip-limited propulsion; brake opposes forward motion | Tire forces |
| Support | Fixed height, yaw-only pose | Spring/damper support |
| Walls | Rotated box footprint clamped inside the enclosure; velocity stopped on collision | Newton contacts |
| Integration | One Warp kernel per physics substep | Tire kernel, collision pipeline, XPBD solver |
| Sensors/viewer | Existing 3D LiDAR and Viser playback | Same |

Lean uses an implicit lateral/yaw force prediction to handle stiffness at low
speeds, then clips forces to the available grip. It is an illustrative racing
model, not calibrated F1TENTH dynamics. It has no suspension, roll/pitch,
free fall, wheel spin, Ackermann linkage, reverse throttle, or terrain following.
Terrain height/tilt can be added separately later. Newton still provides the
earlier free-body checks. The fixed lean height uses the original nominal ride
height; spring/damper and track-width parameters otherwise do not affect lean
dynamics. The lean collision flag describes the latest substep.

Python's `Simulation` keeps `engine="newton"` as its default for compatibility.
Pass `engine="lean"` explicitly. Driving CLI demos default to lean.

## Control the car

```python
from warptracer.simulation import Simulation

sim = Simulation(scenario="drive", engine="lean", device="cpu")
sim.set_action(throttle=0.4, brake=0.0, steering=0.15)
for _ in range(240):
    sim.step()  # advances one physics timestep; inputs persist

sim.set_action(brake=1.0)  # brake without commanding reverse
for _ in range(240):
    sim.step()
```

Throttle and brake are clamped to [0, 1]; steering is in radians, positive left.
A convenience controller accepts `sim.set_target_speed(1.0, steering=0.15)`.
It calculates throttle/brake from forward-speed error on the simulation device.
It is a proportional controller, so drag can leave a small steady speed error.
A zero target applies the brake. This is the interface a later disparity-extender
driver can use to request speed and steering.

For a recorded run, provide a callback that sets controls each step:

```python
def driver(sim, time):
    sim.set_action(throttle=0.5 if time < 1 else 0,
                   brake=1.0 if time >= 1 else 0)

trajectory = sim.run(duration=4, controller=driver)
```

`run()` starts from reset; it does not continue an earlier manual run. `reset()`
clears controls and steering state. Without a callback, named driving scenarios
use their scripted speed/steering requests; `drive` stays neutral. Headless
physics makes no per-step state copies to the CPU. The demo loop launches each physics step. Use the transition runner below for
graph execution; parallel environments are a later milestone.

`DriveConfig` in `driving.py` holds the force parameters, `Track` and `Vehicle`
in `scene.py` hold dimensions/mass, `simulation.py` advances physics, and
`playback.py` exports the viewer. Units are meters, kilograms, seconds, and radians.
Coordinates are Z up, X forward, Y left. Saved poses are `x,y,z,qx,qy,qz,qw`,
velocities are `vx,vy,vz,wx,wy,wz`, and applied controls are
`throttle,brake,steering_rad`. The initial sample has zero applied controls;
later samples store the controls used in the preceding physics step.
Newton's world-frame force/torque convention is documented
[here](https://newton-physics.github.io/newton/stable/concepts/conventions.html).

## Single-car benchmark and graph execution

Run on your local GPU or in the benchmark Colab notebook:

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --physics lean --backend both --output outputs/lean-benchmark.json
uv run --locked python -m warptracer.benchmark --device cuda:0 --physics newton --backend both --output outputs/newton-benchmark.json
```

Each command compares physics only, physics plus LiDAR, and physics plus LiDAR
and host recording. The benchmark uses **one car**, four 240 Hz substeps per
transition, **1080 rays at 60 Hz**, and host recording at 30 Hz. The demo uses
30 Hz LiDAR, so its timing is a different workload. The 200k aggregate transitions/s
reported for a batched simulator cannot be compared directly to single-car
physics substeps/s.

Five trials per case follow two simulated seconds of warmup. Timing excludes
compilation, construction, reset, parity validation, HTML export, and disk writes.
Recording includes host copies and retained samples, but no HTML export.
The report contains all trials, medians, hardware/software, final states,
physics substeps/s, and environment transitions/s. CUDA event intervals include
stream idle gaps and are not summed kernel durations. `--profile` saves a separate
Python call profile; it does not profile CUDA kernels.

Graph execution is checked against eager state, controls, and scans before timing.
Capture warms the used kernels and solver allocations first. Manual inputs remain
in device buffers and can change between transitions:

```python
from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation

sim = Simulation(scenario="drive", engine="lean", device="cuda:0",
                 lidar=LidarConfig(frequency=60))
runner = TransitionRunner(sim, backend="graph", substeps=4)
sim.set_target_speed(1, steering=.2)
for _ in range(60):
    runner.advance()
runner.reset()
```

Use the runner exclusively for stepping and reset while it owns the simulation.
An even substep count restores the captured input/output buffer identities;
when LiDAR is enabled its rate must equal the transition rate. A device-side
scripted circle controller is used for benchmarking, avoiding host input updates
during the timed loop. Ordinary Python controllers can still set inputs between
transitions.

Without a GPU, use `--device cpu --backend both`. Warp's CPU API capture can test
replay correctness and reduce Python overhead, but its timing is not a CUDA
measurement. This environment has no CUDA driver; native CUDA behavior needs
validation on your GPU. Automatic backend selection uses both on CUDA and eager
on CPU.

## Verification and next milestone

```bash
uv run --locked --extra viz --extra dev pytest -q
```

See [VALIDATION.md](VALIDATION.md) for results and limits. Existing root dependencies
and `uv.lock` are unchanged. Next: run the GPU benchmark, then batch the lean
state/control arrays across environments. Disparity extender can use the existing
LiDAR and target-speed/steering interface; PPO is not added here.
