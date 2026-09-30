# RaceSense3D

A standalone first milestone for a 3D autonomous-racing simulator: verified
triangle-mesh ray casting, LiDAR and ideal pinhole depth. No vehicle dynamics,
RL environment, ROS, graphics context, or dependency on WarpORacer.

## Run in Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/yuywe/WarpTracer/blob/sensor-prototype/racesense3d/notebooks/colab_sensors.ipynb)
and run its three code cells: clone `sensor-prototype`, install the package, then
run the geometry tests and sensor demo. Source stays in GitHub; the notebook
contains no embedded archive. A CPU runtime works. Select a GPU runtime to run
CUDA tests if Warp detects it. Hosted Colab execution remains unverified.

## Local use

Python 3.10+:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/sensors.py sensor_preview.png
```

```python
import numpy as np
from racesense3d import demo_scene, lidar, camera, CAMERA_TO_BODY

scene = demo_scene(device='cpu')  # or 'cuda:0'; omitted = auto
laser = scene.sensor(lidar(), batch_size=2)
scan = laser.scan([[0,0,1], [1,0,1]])
ranges, valid = scan.numpy()       # (2, 1, 1080)

camera_sensor = scene.sensor(camera(320,240), batch_size=1)
depth = camera_sensor.scan([[0,0,1]], CAMERA_TO_BODY)
values, valid = depth.numpy()      # (1, 240, 320)
```

## Measurement contract

- Geometry and measurements use meters; angles use radians except `hfov_degrees`.
- World/body/LiDAR: right-handed X forward, Y left, Z up.
- Camera optical: X right, Y down, Z forward. `CAMERA_TO_BODY` maps optical
  vectors into the body frame. All supplied rotations map sensor-local to world.
- Compose mounts as `R_world_sensor = R_world_body @ R_body_sensor` and
  `p_world_sensor = p_world_body + R_world_body @ p_body_sensor`.
- LiDAR returns Euclidean ray distance. Camera returns optical-axis Z depth.
  Calibrated `fx`, `fy`, `cx`, `cy` are supported; default pixels are square.
- Measurement interval is **near <= value < far**. Misses and out-of-range
  first hits return `far` with validity false. A too-near object still occludes.
- Triangles are opaque and two-sided. Closest positive intersections are used.
  Sensors should not be placed exactly on geometry: surface-origin behavior is
  numerically ambiguous. Self-hit filtering is not implemented.
- LiDAR layouts are `(batch, elevation, azimuth)`; camera `(batch, H, W)`.
  Full-360 azimuth arrays should omit a duplicate endpoint.
- `Scan.points(...)` reconstructs world XYZ; invalid points become NaN.

## Batching and memory

One immutable `Scene` holds a BVH-backed triangle mesh. `Sensor` caches rays,
pose buffers and output buffers. Its kernel parallelizes over `(batch, ray)`.
Each batch member sees the same mesh from its own pose; there are no interacting
cars or different maps per member yet. Scene geometry can be arbitrary triangles
via `Scene(vertices, triangles)`; primitive helpers make test fixtures.

`scan()` validates and uploads host poses. `scan_device()` accepts existing Warp
vec3/mat33 pose arrays without host transfer; the caller ensures finite positions,
proper rotations, array lifetime and stream ordering. Results remain resident
until `.numpy()` or `.points()` is requested. Output buffers are **borrowed and
reused**: a subsequent scan overwrites earlier results from that Sensor. Copy if
retention is needed. A sensor instance is not thread-safe. Torch stream integration
and zero-copy wrappers are not implemented by this milestone.

## Verification

The suite checks analytic walls, oblique ranges, axial camera depth/clipping,
calibrated reprojection, floor/elevation orientation, near occlusion, two-sided
triangles, range boundaries, translated batches, arbitrary rigid transformations,
resident input buffers, reuse, and bad inputs. An independent scalar ray–box slab
reference checks 4,346 ray/pose combinations, including origins inside the box
and axis-parallel rays. CPU/CUDA agreement is tested when CUDA is detected.

See `VALIDATION.md` for the actual environment and results from this build.

## Scope and next development

This is a sensor subsystem, not yet a Gymnasium racing environment. No physics,
contact response, noise, distortion, rolling shutter, LiDAR motion distortion,
materials, stereo matching or dynamic objects. Depth is ideal geometric depth,
not a full simulation of a RealSense stereo pipeline. Visualization is optional.

Suggested next steps after sensor validation: scene import and realistic track
geometry, timed sensor poses, then vehicle dynamics and a Gymnasium wrapper.
Keep physics and sensors coupled through poses rather than embedding dynamics
inside the ray-casting kernel.

Reference inspected: https://github.com/uci-f1tenth/warporacer at
`e16e4473cb86e6ae2e1fa1b078e5a1e04b20d7f2`. It uses 2D distance-field ray marching;
this project independently implements 3D mesh queries and copies no source or
map assets from that repository. NVIDIA Warp: https://github.com/NVIDIA/warp.
