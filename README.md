# WarpTracer

A minimal racing simulator with a box car and vehicle-mounted LiDAR. Choose a
flat room or an oval road with gentle elevation changes. The car can accelerate,
brake, and steer. A reactive
disparity controller navigates from LiDAR; Warp’s built-in OpenGL renderer displays replays.

Work is on the `performance-prototype` branch. The default lean physics model
follows the road surface; the room remains flat. Newton is available for the
original room experiments.
Batched cars run independently and cannot see or collide with each other.
PPO can train a policy from LiDAR and vehicle state on the elevated oval or a PNG occupancy map.

## Get started

Install Git and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone --branch performance-prototype https://github.com/yuywe/WarpTracer.git
cd WarpTracer
uv run warptracer demo --view
```

For an existing checkout, switch to `performance-prototype` and run
`git pull --ff-only` before the demo. Run commands from the repository root;
uv installs dependencies and uses one environment there. The first simulation
run compiles kernels.

The demo runs **30 simulated seconds of LiDAR navigation**, using lean physics,
108 beams at 60 Hz, graph execution and fused substeps. It selects CUDA when
available and otherwise uses CPU. Replay dependencies are included automatically.

With `--view`, the recorded run opens in Warp’s native window. Use Space to pause,
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
| Acceleration and braking check | `uv run warptracer demo accelerate-brake` | `outputs/accelerate-brake.npz` |
| Circle driving check | `uv run warptracer demo circle` | `outputs/circle.npz` |
| Quick performance check | `uv run warptracer benchmark` | `outputs/quick.json` |
| LiDAR vs. disparity navigation | `uv run warptracer benchmark navigation` | `outputs/navigation.json` |
| Navigation on the elevated oval | `uv run warptracer benchmark oval` | `outputs/oval-benchmark.json` |
| Mesh vs. EDT LiDAR | `uv run warptracer benchmark grid` | `outputs/grid.json` |
| Large batches and moving LiDAR (up to 4,096 cars) | `uv run warptracer benchmark scale` | `outputs/scale.json` |
| Independent car scaling | `uv run warptracer benchmark batches` | `outputs/batches.json` |
| Fused vs. unfused physics | `uv run warptracer benchmark fusion` | `outputs/fusion.json` |
| PPO training on the oval (1,024 cars, CUDA) | `uv run --extra rl warptracer train oval` | `outputs/ppo/checkpoint.pt` |
| Prepare a PNG map once (set its meters/pixel) | `uv run warptracer map maps/my-track.png --resolution 0.05` | `outputs/maps/my-track.png` |
| PPO on your PNG map | `uv run --extra rl warptracer train oval --map maps/my-track.png` | `outputs/ppo/checkpoint.pt` |
| Evaluate and view a trained policy | `uv run --extra rl warptracer evaluate --view` | `outputs/ppo/evaluation.npz` |
| Render an MP4 for Colab | `uv run warptracer demo oval-grid --video` | `outputs/oval-grid.mp4` |

The `scale`, `grid`, `oval`, `navigation`, `batches` and `fusion` presets require CUDA by default.
They fail immediately if a GPU is unavailable. The quick benchmark also works
on CPU. See [benchmark presets and result units](docs/benchmarking.md).

The oval demo runs for 120 simulated seconds on a 16 × 10 m loop with a nominal
2 m lane and 0.4 m elevation range. The chassis follows road height and tilt;
LiDAR and replay use the same 3D track mesh. View `outputs/oval.npz` with
`uv run warptracer view outputs/oval.npz` and check
the printed lap/contact counts. See [tracks and terrain following](docs/tracks.md).

The optional `oval-grid` demo marches full 3D rays using precomputed EDT
clearance and terrain height. It creates no sensor triangle mesh. The mesh backend
remains the default: grid boundaries approximate the track. The T4 comparison showed a 16%
navigation throughput gain over mesh at 256 cars; larger batches and the optional
sample cache are measured separately by `scale`. See [the comparison workflow and accuracy limits](docs/grid-lidar.md).

Only add an option when you need to change something:

```bash
uv run warptracer demo --seconds 60
```

Use `uv run warptracer demo --help` or `uv run warptracer benchmark --help`
for advanced settings. Existing `warptracer-demo` and Python module commands
remain available.

## Colab

Select **Runtime → Change runtime type → GPU**, then open a notebook:

- [Driving replay](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/rigidbody_colab.ipynb): run setup, simulate and render with Warp, then play the MP4. It defaults to `scenario = "oval-grid"`; select `"oval"` for mesh sensing.
- [Benchmarks](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/benchmark_colab.ipynb): run setup, choose a `preset`, then read the results. It defaults to `preset = "scale"` for moving scans, sample-cache comparisons and navigation with up to 4,096 cars. Use `"grid"` to repeat the earlier fixed-scan comparison.
- [PPO training](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/ppo_colab.ipynb): run setup, train with `preset = "oval"`, inspect learning, then evaluate, display the Warp-rendered MP4, and download the checkpoint/recording.

All reuse the same checkout and root uv environment. Output streams into the
cell. No command construction or physics settings need editing for normal runs.

## Details and development

- [Navigation behavior and limits](docs/navigation.md)
- [Tracks and terrain following](docs/tracks.md)
- [Grid LiDAR experiment, accuracy, and Colab comparison](docs/grid-lidar.md)
- [Benchmark presets, timing, and batch API](docs/benchmarking.md)
- [PPO presets, episodes, checkpoints and reference review](docs/ppo.md)
- [PNG map import, route preview and Colab upload](docs/png-maps.md)
- [Warp native playback and Colab videos](docs/playback.md)
- [Vehicle, LiDAR, controls, and saved arrays](docs/reference.md)
- [Validation results and model limits](VALIDATION.md)

Run tests with `uv run --extra dev pytest -q`.
Include PPO tests with `uv run --extra dev --extra rl pytest -q`.
