# WarpTracer

A minimal GPU racing simulator built on NVIDIA Warp: box cars with vehicle-mounted
LiDAR, driven by a reactive disparity controller or a PPO policy. Cars run on a flat
room, an elevated oval, a PNG occupancy map, or a two-layer figure eight that crosses
over itself. Physics, sensing and PPO rollouts run on CUDA, with thousands of
independent cars in a batch; everything also works on CPU for small checks. Cars
cannot see or collide with each other.

Work is on the `performance-prototype` branch.

## Get started

Install Git and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone --branch performance-prototype https://github.com/yuywe/WarpTracer.git
cd WarpTracer
uv run warptracer demo --view
```

Run commands from the repository root; uv installs dependencies and uses one
environment there. The first simulation run compiles kernels.

The demo runs **30 simulated seconds of LiDAR navigation** with 108 beams at 60 Hz.
With `--view`, the recorded run opens in Warp's native window. Use Space to pause,
Left/Right to seek, R to restart, mouse drag to look and WASD to move the camera.
The `.npz` embeds the scene, poses, controls and scans; `.json` saves the settings.
To reopen it, run `uv run warptracer view outputs/disparity.npz`.
Viewing uses NVIDIA CUDA/OpenGL; simulation and recording also work on CPU.
For Colab, use `--video` to export an MP4. [Viewing and video details](docs/playback.md).

## Common commands

| What you want | Command | Output |
| --- | --- | --- |
| LiDAR navigation recording | `uv run warptracer demo` | `outputs/disparity.npz` |
| Elevated oval with native playback | `uv run warptracer demo oval --view` | `outputs/oval.npz` |
| Oval with EDT-based 3D LiDAR | `uv run warptracer demo oval-grid` | `outputs/oval-grid.npz` |
| Under and over a bridge | `uv run warptracer demo bridge` | `outputs/bridge.npz` |
| Acceleration and braking check | `uv run warptracer demo accelerate-brake` | `outputs/accelerate-brake.npz` |
| Quick performance check | `uv run warptracer benchmark` | `outputs/quick.json` |
| Other benchmarks (CUDA): `navigation`, `oval`, `grid`, `batches`, `fusion` | `uv run warptracer benchmark oval` | `outputs/oval-benchmark.json` |
| PPO on the oval (1,024 cars, CUDA) | `uv run warptracer train oval` | `outputs/ppo/checkpoint.pt` |
| PPO on the bridge figure eight | `uv run warptracer train oval --bridge` | `outputs/ppo/checkpoint.pt` |
| Prepare a PNG map once (set its meters/pixel) | `uv run warptracer map maps/my-track.png --resolution 0.05` | `outputs/maps/my-track.png` |
| PPO on your PNG map | `uv run warptracer train oval --map maps/my-track.png` | `outputs/ppo/checkpoint.pt` |
| Evaluate and view a trained policy | `uv run warptracer evaluate --view` | `outputs/ppo/evaluation.npz` |
| Render an MP4 for Colab | `uv run warptracer demo oval-grid --video` | `outputs/oval-grid.mp4` |

Use `--help` on any command for advanced settings; add an option only when you need
to change something, such as `uv run warptracer demo --seconds 60`.

## Colab

Select **Runtime → Change runtime type → GPU**, then open a notebook:

- [Driving replay](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/rigidbody_colab.ipynb): run setup, simulate and render with Warp, then play the MP4. It defaults to `scenario = "oval-grid"`.
- [PPO training](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/ppo_colab.ipynb): run setup, train with `preset = "oval"`, inspect learning, then evaluate, display the Warp-rendered MP4, and download the checkpoint/recording.

## Details and development

- [Navigation behavior and limits](docs/navigation.md)
- [Tracks and terrain following](docs/tracks.md)
- [Layered tracks](docs/layers.md)
- [Grid LiDAR, accuracy and limits](docs/grid-lidar.md)
- [Benchmark presets, timing, and batch API](docs/benchmarking.md)
- [PPO presets, episodes, checkpoints and reference review](docs/ppo.md)
- [PNG map import, route preview and Colab upload](docs/png-maps.md)
- [Warp native playback and Colab videos](docs/playback.md)
- [Vehicle, LiDAR, controls, and saved arrays](docs/reference.md)
- [Validation results and model limits](VALIDATION.md)

Run tests, including PPO, with `uv run --extra dev pytest -q`.
