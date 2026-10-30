# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

WarpTracer is a minimal GPU-accelerated racing simulator built on NVIDIA Warp
(and optionally Newton physics): a box car with vehicle-mounted LiDAR, driving in
a flat room or on an elevated oval track, navigated by a reactive disparity-extender
controller (or a PPO policy trained on the oval or a PNG occupancy map). Replays
use Warp's native OpenGL viewer or an offscreen MP4. Active work is on the
`performance-prototype` branch (not `main`).

## Commands

Run everything from the repository root; `uv` manages the single root environment
(no nested project, no second venv).

```bash
uv run warptracer demo                      # 30s disparity navigation -> outputs/disparity.npz (+ .json)
uv run warptracer demo oval                 # elevated oval -> outputs/oval.npz
uv run warptracer demo oval-grid --view     # oval with grid LiDAR, then open the native Warp viewer
uv run warptracer demo oval --video         # render outputs/oval.mp4 offscreen (Colab)
uv run warptracer demo oval --follow        # top-down camera tracking the car (optional altitude, default 20 m)
uv run warptracer demo --seconds 60         # override one setting; see --help for the rest
uv run warptracer demo bridge               # two-layer figure eight: under, then over a 0.7 m bridge
uv run warptracer demo --help               # scenarios: oval, oval-grid, bridge, disparity, accelerate-brake,
                                            #   circle, s-turn, drop, wall-impact

uv run warptracer view outputs/oval.npz     # reopen a recording (add --video to render an MP4 instead)

uv run warptracer map maps/my-track.png --resolution 0.05   # create a JSON sidecar + route preview
uv run warptracer train smoke               # PPO path check on CPU; presets: smoke, oval, large
uv run warptracer train oval --map maps/my-track.png        # train on a PNG map (oval/large need CUDA)
uv run warptracer train oval --bridge                       # train on the two-layer bridge figure eight
uv run warptracer evaluate --view           # evaluate outputs/ppo/checkpoint.pt (or --video)

uv run warptracer benchmark                 # quick preset (CPU-friendly) -> outputs/quick.json
uv run warptracer benchmark navigation      # LiDAR vs. disparity-control throughput; requires CUDA
uv run warptracer benchmark oval            # navigation on the elevated oval; requires CUDA
uv run warptracer benchmark batches         # independent-car scaling; requires CUDA
uv run warptracer benchmark fusion          # fused vs. unfused physics; requires CUDA
uv run warptracer benchmark --help

uv run --extra dev pytest -q                # run the full test suite
uv run --extra dev pytest tests/test_disparity.py -q         # single test file
uv run --extra dev pytest tests/test_disparity.py::test_name -q  # single test
```

The `oval`, `navigation`, `batches`, and `fusion` presets fail immediately without
a CUDA GPU; only `quick` (default) runs on CPU. First run compiles Warp kernels.
Without `--view`/`--video`, demos and evaluation only save the recording; no
graphics context is created. The viewer needs an NVIDIA CUDA driver and an
OpenGL context (Warp 1.17 allocates CUDA pinned memory even in its fallback), so
it can't run on a CPU-only host, though simulating and recording still work there.
Older module-level entry points (`warptracer-demo`, `python -m warptracer.*`) still
work for lower-level/custom experiments.

## Architecture

**Engines.** `Simulation` (`simulation.py`) supports two physics backends, selected
via `engine=`:
- `"lean"` — a custom planar bicycle model (`lean.py`) running as hand-written Warp
  kernels. Supports batched, independent cars (`num_envs=N`), the elevated oval
  (terrain-following height/tilt), and fused multi-substep kernels for graph capture.
  This is the actively developed path and what the CLI demos/benchmarks use.
- `"newton"` — the original six-DOF rigid-body model using the `newton` physics
  library (`scene.py:build_model`, XPBD solver). Room-only, single car, kept for
  compatibility/baseline comparisons. `Simulation`'s Python default is still
  `engine="newton"` for backward compatibility, but CLI demos default to `"lean"`.

**Execution model.** `TransitionRunner` (`execution.py`) owns stepping once created
— don't interleave `sim.step()`/`sim.reset()` with an active runner. A "transition"
is a fixed bundle of physics substeps (even count, required for state-buffer parity)
plus one LiDAR scan at the end. Two independent axes combine:
- `backend`: `"eager"` (Python submits each op) vs. `"graph"` (Warp CUDA/CPU-API
  graph captured once, replayed via `wp.capture_launch`).
- `integrator`: `"unfused"` (one kernel launch per substep) vs. `"fused"` (all
  substeps of a transition in one kernel; lean physics only).
Every fused/graph variant is validated bit-for-bit (within tolerance) against
unfused eager execution before being trusted for timing — see
`benchmark.py:validate_pair`.

**Driving stack.** `scene.py` holds static geometry (`Track`, `Vehicle` dataclasses);
`terrain.py` (`OvalTrack`) extends `Track` with the elevated-loop analytic surface
shared by physics and the LiDAR/playback mesh; `driving.py` holds `DriveConfig`
(tire/steering force parameters) and scripted driving scenarios; `lean.py` has the
bicycle integration kernels (`integrate_bicycle`, `integrate_bicycle_fused`).
Coordinates are Z-up, X-forward, Y-left; poses are `x,y,z,qx,qy,qz,qw`.

**Sensing.** `lidar.py` (`MountedLidar`, `LidarConfig`) adapts the vendored ray
caster in `src/racesense3d/` (copied unchanged from an earlier `sensor-prototype`
commit — treat as a stable third-party-style dependency, not code to casually
refactor) to the moving vehicle/track mesh. Scans run on the simulation device;
sensing adds no physics forces. The host vehicle is excluded from its own
ray-casting mesh.

**Navigation.** `disparity.py` (`DisparityController`) implements a range-disparity
extender (Otterness-style) reading the latest LiDAR scan and writing throttle/brake/
steering directly into the device command buffer each transition. Room driving has
an extra corner-latch heuristic (disabled on the oval, which instead converts the
selected ray to a bounded-lookahead bicycle steering angle). This is a prototype
heuristic controller, not a collision guarantee or a trained policy — no map,
localization, RL, or reversing/recovery behavior.

**Batches.** `num_envs=N` with `engine="lean"` runs N fully independent cars
(shared static track geometry, separate everything else — state, scans, controller
state). Cars cannot see or collide with each other. Single-car code (`num_envs=1`)
is the N=1 special case of the same APIs.

**Recording/replay.** `simulation.py:Simulation.run()` drives either a Python
callback controller or a `TransitionRunner`, from a fresh `reset()`, and returns a
`Trajectory` (poses, velocities, actions, optional `LidarRecording`).
`playback.py:save_recording` writes a self-contained `.npz` (numeric arrays plus
embedded versioned JSON scene/metadata strings, no pickle) and `demo.py` also
writes a `.json` metadata sidecar. `playback.py:play` replays it with
`warp.render.OpenGLRenderer` (pyglet window; or offscreen frames encoded to MP4
via ImageIO/FFmpeg). There is no Viser or HTML exporter anymore. Playback only
replays recorded motion at 30 display FPS and never reruns physics. Rendering
happens after simulation and is excluded from reported throughput.

**Maps and PPO.** `maps.py` (`MapTrack`, `load_map`) imports occupancy PNGs
(white = drivable) as flat tracks with 1 m walls. A PNG needs a same-stem `.json`
sidecar giving meters per pixel. The ROS-style `.yaml` files under `maps/` are not
read by `maps.py`. `train.py` runs batched PPO from LiDAR plus vehicle features
(presets `smoke`/`oval`/`large`). Checkpoints embed the map geometry, so
evaluation and resume restore it without the original files. See `docs/ppo.md` and
`docs/png-maps.md`.

**Layered tracks.** `layers.py` (`LayeredTrack`, `figure_eight_bridge`) stacks 2.5D
height layers (floor + decks) so a road can cross over itself; `layer_kernels.py`
integrates them. Each car carries a layer index and switches only at seams, where
another layer is drivable at the same height. Physics reads grids; LiDAR/replay use
a mesh built from them (mesh LiDAR only). PPO trains on the bridge figure eight
(`train --bridge`); `rl_env.py` treats layered tracks like PNG maps for route
progress and respawns each car on its route sample's layer. See `docs/layers.md`.

**CLI.** `cli.py` is a thin dispatcher (`demo`, `benchmark`, `train`, `evaluate`,
`map`, `view`) to the `main` of `demo.py`, `benchmark.py`, `train.py`, `maps.py`
and `playback.py`. `benchmark.py:PRESETS` centralizes workload/device/env-count/
duration combinations so adding a new benchmark preset is a dict entry plus
`make_runner` wiring, not new argument-parsing code.

## Working in this codebase

- Units throughout are meters, kilograms, seconds, radians.
- When changing lean physics, tire forces, or terrain, re-run the relevant
  parity/validation tests (`test_lean.py`, `test_oval.py`, `test_physics.py`,
  `test_benchmark.py`'s `validate_pair` path) — fused/graph paths are only trusted
  because they're checked against unfused eager execution.
- `docs/` has the authoritative behavioral detail for vehicle/sensor parameters
  (`docs/reference.md`), the disparity controller (`docs/navigation.md`), the oval
  track (`docs/tracks.md`), benchmark presets/units (`docs/benchmarking.md`),
playback/video (`docs/playback.md`), grid LiDAR (`docs/grid-lidar.md`), layered tracks (`docs/layers.md`), PNG maps
(`docs/png-maps.md`) and PPO (`docs/ppo.md`).
  Prefer updating these alongside behavior changes rather than duplicating detail
  in code comments.
- `VALIDATION.md` records point-in-time test/behavior validation runs (including
  GPU results that can't be reproduced in a CPU-only environment) — treat it as a
  changelog of verification evidence, not living documentation.
- There is an untracked top-level `racesense3d/` directory (a separate nested
  checkout with its own `.venv`) alongside the tracked `src/racesense3d/` package —
  these are different things; don't confuse them, and don't assume the top-level
  one needs to stay in sync with anything.
