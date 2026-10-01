# WarpTracer — rigid-body playground

A flat oval track and a vehicle-shaped rigid body, simulated with Newton.
The `rigidbody-prototype` branch is the next small experiment after the sensor
prototype: gravity, floor contact, barrier contact, and interactive 3D playback.

[Open in Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/rigidbody-prototype/notebooks/rigidbody_colab.ipynb)

## Run in Colab

Run the notebook's three cells: clone/sync, simulate, display the replay.
Colab's existing `uv` creates **one environment at the repository root**.
The notebook uses Colab's Python interpreter. CPU works; a GPU runtime selects
CUDA automatically when Warp detects it. First execution includes kernel compilation.

The embedded replay lets you orbit, zoom, pause, and scrub through time.
The motion is recorded: moving the viewing camera does not affect the physics.
The notebook runs the scene demo; it does not run the sensor verification suite.

## Run with uv

From the repository root:

```bash
uv sync --locked --extra viz
uv run --locked --extra viz warptracer-demo
```

Open `outputs/drop.html` in a browser. The file includes the viewer and recording;
no persistent Python server is needed for playback.

Two experiments are included:

```bash
# A slightly tilted chassis falls, slides a little, and settles.
uv run --locked --extra viz warptracer-demo --scenario drop

# An initial sideways velocity sends the chassis into the outer barrier.
uv run --locked --extra viz warptracer-demo --scenario wall-impact

# Run physics without the viewer or intermediate pose copies.
uv run --locked warptracer-demo --headless --device cpu
```

Options: `--device cpu|cuda:0`, `--seconds 4`, `--physics-hz 240`,
`--record-fps 30`, and `--output outputs`. The recording frequency must divide
the physics frequency. `python main.py` is also available through `uv run`.

Each run writes pose/velocity data (`.npz`) and configuration/timing (`.json`).
Recorded runs additionally write `.html`. Headless artifacts use a `_headless`
suffix so they do not overwrite a recorded run's data. Outputs are ignored by Git.

## What is modeled

| Part | Representation |
| --- | --- |
| Ground | Static infinite collision plane; a finite ground patch is displayed |
| Oval | 6 m straights, 2.5 m centerline bend radius, 1.6 m nominal lane width |
| Barriers | Fixed box colliders, 0.3 m high and 0.12 m thick, with slightly overlapping segments |
| Chassis | One free rigid box, 0.52 × 0.26 × 0.12 m, 3.2 kg |
| Vehicle detail | Visual deck, front marker, and tire-like blocks attached to the chassis |
| Physics | Newton XPBD, 240 Hz, eight solver iterations, gravity and friction |
| Playback | Viser/WebGL; geometry plus sampled transforms, 30 frames/s by default |

The boundary centerlines define the nominal lane width; barrier thickness reduces
the clear driving width. Physics and playback use the same barrier dimensions
and transforms. Colors and road markings are visual.

Coordinates are meters with Z up, X forward, Y left. Saved poses use
`x,y,z,qx,qy,qz,qw`; saved velocities use `vx,vy,vz,wx,wy,wz`.
The playback adapter converts quaternions to Viser's WXYZ ordering.

This milestone uses the chassis box for contact. The tire-like blocks do not
rotate, steer, or generate traction. The car is not driving around the oval yet.
Dimensions, mass, and friction are illustrative defaults, not measured vehicle data.

## Change the scene

The small public building blocks are `Track`, `Vehicle`, and `Simulation`:

```python
from warptracer.scene import Track, Vehicle
from warptracer.simulation import Simulation

sim = Simulation(
    track=Track(straight_length=8.0, lane_width=2.0),
    vehicle=Vehicle(mass=3.5),
    scenario="drop",
    device="cpu",
)
trajectory = sim.run(duration=4.0)
```

`src/warptracer/scene.py` builds geometry, `simulation.py` advances physics,
and `playback.py` exports visualization. The physics module does not import Viser.
Newton and Viser are pinned in `pyproject.toml`; the root `uv.lock` captures the
resolved dependencies. The larger Newton example/RTX bundles are not required.

## Verification

```bash
uv sync --locked --extra viz --extra dev
uv run --locked --extra viz --extra dev pytest -q
```

See [VALIDATION.md](VALIDATION.md) for measured behavior and verification limits.
The existing `verify_raycasts.py` and `lidar_verification.png` are earlier
repository experiments; this demo does not invoke them. The separate
[`sensor-prototype` branch](https://github.com/yuywe/WarpTracer/tree/sensor-prototype)
contains the sensor package and its notebook.
