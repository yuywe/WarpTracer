# Vehicle and sensor reference

[Back to the quick start](../README.md) · [Benchmark guide](benchmarking.md)

## Vehicle-mounted LiDAR

The four Python library files in `src/racesense3d/` are copied **unchanged** from
[`sensor-prototype` commit 51b2d1c](https://github.com/yuywe/WarpTracer/tree/51b2d1c8a4c4460d1b1828e5b0240ee287ba577d/racesense3d/src/racesense3d).
`warptracer/lidar.py` adapts that reusable ray caster to the vehicle and track.
Both packages use the existing root uv project; there is no nested project or
second environment. Only LiDAR is instantiated; camera code is not used.

| Setting | Default |
| --- | --- |
| Beam layout | 108 beams in one horizontal ring, -135° to +135°, ordered right to left |
| Range | 0.021 m minimum, 30 m maximum |
| Rate | 30 Hz, independently of pose recording rate |
| Mount translation | 0.12 m forward, centered laterally, 0.025 m above the chassis top |
| Mount orientation | Aligned with the chassis; follows its yaw, pitch, and roll |
| Scanned surfaces | The same four wall boxes and floor as the physics scene |
| Host vehicle | Excluded from the ray-casting mesh |
| Replay | Up to 180 returns and 30 ray lines; all beams are saved |

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
print(scans.ranges.shape)   # (301, 108): initial scan plus 30 scans/s
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
| Integration | One Warp kernel per substep, or fused substeps via TransitionRunner | Tire kernel, collision pipeline, XPBD solver |
| Sensors/viewer | Existing 3D LiDAR and Viser playback | Same |

Lean uses an implicit lateral/yaw force prediction to handle stiffness at low
speeds, then clips forces to the available grip. It is an illustrative racing
model, not calibrated F1TENTH dynamics. It has no suspension, roll/pitch,
free fall, wheel spin, Ackermann linkage, reverse throttle, or terrain following.
Terrain height/tilt can be added separately later. Newton still provides the
earlier free-body checks. The fixed lean height uses the original nominal ride
height; spring/damper and track-width parameters otherwise do not affect lean
dynamics. The lean collision flag describes the latest substep.
`wall_contact_substeps` accumulates every contacting substep, including inside
fused transitions, and clears on reset. It counts contact duration in substeps,
not distinct collisions. See [LiDAR navigation](navigation.md) for the controller.

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
physics makes no per-step state copies to the CPU. The demo loop launches each physics step. Use the [transition runner](benchmarking.md) for graph execution; batched
environments are available through that runner.

`DriveConfig` in `driving.py` holds the force parameters, `Track` and `Vehicle`
in `scene.py` hold dimensions/mass, `simulation.py` advances physics, and
`playback.py` exports the viewer. Units are meters, kilograms, seconds, and radians.
Coordinates are Z up, X forward, Y left. Saved poses are `x,y,z,qx,qy,qz,qw`,
velocities are `vx,vy,vz,wx,wy,wz`, and applied controls are
`throttle,brake,steering_rad`. The initial sample has zero applied controls;
later samples store the controls used in the preceding physics step.
Newton's world-frame force/torque convention is documented
[here](https://newton-physics.github.io/newton/stable/concepts/conventions.html).


## Batches

Pass `num_envs=N` with `engine="lean"` for independent cars sharing static track
geometry. The single-car API and array shapes are unchanged when N=1.
For batch controls, captured stepping, array shapes, and resets, see the
[benchmark API example](benchmarking.md#python-api).
