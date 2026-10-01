# WarpTracer — vehicle controls

A box car that accelerates, brakes, and steers on a flat surface with four walls.
The `vehicle-controls` branch builds on the minimal `rigidbody-prototype` scene.
Newton advances the rigid body; four invisible tire contact points apply forces.
The scene still has **one moving body and six visible shapes**.

[Open in Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/vehicle-controls/notebooks/rigidbody_colab.ipynb)

## Run in Colab

Run the three cells: setup, simulate, display. Setup reuses the earlier
`/content/WarpTracer-rigidbody` clone, switches to `vehicle-controls`, and updates
it with a fast-forward pull. Colab's preinstalled `uv` uses **one environment at
that repository root** and Colab's Python interpreter. Local edits that conflict
with switching/pulling must be resolved; setup does not discard them.

Choose `accelerate-brake`, `circle`, or `s-turn` in the second cell. Each driving
recording lasts ten seconds by default. Rerun the last two cells after changing
scenarios. CPU works; CUDA is selected automatically when available. The first
execution compiles Warp kernels.

The replay has orbit/zoom controls and a timeline. It contains recorded motion;
camera movement does not rerun physics or control the car.

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
Driving requires at least 120 physics steps/s; recording frequency must divide
physics frequency. Use the default 240 Hz for the supplied spring parameters.

Each run writes pose, velocity, and applied control arrays (`.npz`) and
configuration/timing (`.json`). Recorded runs also write `.html`. Headless runs
sample only the initial and final state, and use a `_headless` filename suffix.
Outputs are ignored by Git.

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

Next: attach the LiDAR from the separate
[`sensor-prototype` branch](https://github.com/yuywe/WarpTracer/tree/sensor-prototype),
then use disparity extender to test autonomous driving. This branch does not run
sensor verification, PPO, or other RL training.
