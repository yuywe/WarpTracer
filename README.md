# WarpTracer — vehicle-mounted LiDAR

A box car that accelerates, brakes, and steers on a flat surface with four walls.
The `vehicle-lidar` branch adds the existing sensor prototype's LiDAR to
`vehicle-controls`. Newton advances the rigid body; four invisible tire contact
points apply forces. There is still **one moving body and six solid scene shapes**,
plus scan points and a sparse ray overlay. The same scripted commands drive the car;
LiDAR is observing, with autonomous navigation coming later.

[Open in Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/vehicle-lidar/notebooks/rigidbody_colab.ipynb)

## Run in Colab

Run the three cells: setup, simulate, display. Setup reuses the earlier
`/content/WarpTracer-rigidbody` clone, switches to `vehicle-lidar`, and updates
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
Use the default 240 Hz for the supplied spring parameters.

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

sim = Simulation(scenario="circle", lidar=LidarConfig(), device="cpu")
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

| Part | Representation |
| --- | --- |
| Car | One rigid box, 0.52 × 0.26 × 0.12 m, 3.2 kg |
| Ground and walls | Static ground plane and four box walls; six total collision shapes including the car |
| Tire contact points | Four mathematical points under the box; 0.32 m wheelbase, 0.22 m track width |
| Support | Spring/damper forces against the flat floor, with no tensile force |
| Acceleration/braking | Tangential forces; braking opposes motion and takes priority over throttle |
| Steering | Front contact directions turn together; ±0.418 rad limit, 1.5 rad/s rate limit |
| Grip | Longitudinal and lateral forces share a friction-circle limit based on each contact's support force |
| Sideways motion | Linear slip-velocity damping capped by available grip |
| Integration | Newton XPBD, 240 Hz, eight iterations; force calculations in a Warp kernel |
| Playback | Viser, with sampled transforms at 30 frames/s |

Driving uses a 12 × 6 m clear area, or 8 × 8 m for the circle demo. The drop and
impact checks keep their 6 × 4 m area. The support points hold the box about
4.3 cm above the floor at rest. This small gap represents the unrendered tires;
no wheel meshes, joints, or extra rigid bodies are added.

This is a deliberately simple **flat-ground** force model. Tire support uses an
analytic z=0 plane, not terrain raycasts. Newton still handles chassis collisions
with the ground and walls, including when the chassis bottoms out. Tires exert
no force when out of reach of the floor or when the body is overturned. The
model has no wheel spin, tire slip-ratio dynamics, Ackermann linkage, reverse
throttle, or calibrated vehicle parameters. It is a foundation for driving and
sensor integration, not a validated racing dynamics model.

## Control the car

```python
from warptracer.simulation import Simulation

sim = Simulation(scenario="drive", device="cpu")
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
physics makes no per-step state copies to the CPU. The Python loop still launches
each physics step; parallel environments and CUDA graph capture are future work.

`DriveConfig` in `driving.py` holds the force parameters, `Track` and `Vehicle`
in `scene.py` hold dimensions/mass, `simulation.py` advances physics, and
`playback.py` exports the viewer. Units are meters, kilograms, seconds, and radians.
Coordinates are Z up, X forward, Y left. Saved poses are `x,y,z,qx,qy,qz,qw`,
velocities are `vx,vy,vz,wx,wy,wz`, and applied controls are
`throttle,brake,steering_rad`. The initial sample has zero applied controls;
later samples store the controls used in the preceding physics step.
Newton's world-frame force/torque convention is documented
[here](https://newton-physics.github.io/newton/stable/concepts/conventions.html).

## Verification and next milestone

```bash
uv run --locked --extra viz --extra dev pytest -q
```

See [VALIDATION.md](VALIDATION.md) for results and limits. Newton, Warp, and Viser
remain pinned in the existing root `uv.lock`; no new dependencies are needed.

Next: connect disparity extender to these scans and the existing target-speed /
steering interface. This branch runs the same scripted driving checks with LiDAR;
it does not add a navigation controller or RL training.
