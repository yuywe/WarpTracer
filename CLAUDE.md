# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

WarpTracer is a minimal GPU-accelerated racing simulator built on NVIDIA Warp
(and optionally Newton physics): a box car with vehicle-mounted LiDAR, driving in
a flat room or on an elevated oval track, navigated by a reactive disparity-extender
controller. Viser exports interactive HTML replays. Active work is on the
`performance-prototype` branch (not `main`).

## Commands

Run everything from the repository root; `uv` manages the single root environment
(no nested project, no second venv).

```bash
uv run warptracer demo                      # 30s disparity-navigation replay -> outputs/disparity.html
uv run warptracer demo oval                 # elevated oval track replay -> outputs/oval.html
uv run warptracer demo accelerate-brake     # -> outputs/accelerate-brake.html
uv run warptracer demo circle               # -> outputs/circle.html
uv run warptracer demo --seconds 60         # override one setting; see --help for the rest
uv run warptracer demo --help

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
`Trajectory` (poses, velocities, actions, optional `LidarRecording`). `demo.py`
serializes this to `.npz` (arrays) + `.json` (metadata) and `playback.py:save_html`
renders the Viser replay — replay scrubbing only replays recorded motion, it does
not rerun physics.

**CLI.** `cli.py` is a thin dispatcher to `demo.py` and `benchmark.py` (`argparse`-based,
one subcommand each). `benchmark.py:PRESETS` centralizes workload/device/env-count/
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
  track (`docs/tracks.md`), and benchmark presets/units (`docs/benchmarking.md`).
  Prefer updating these alongside behavior changes rather than duplicating detail
  in code comments.
- `VALIDATION.md` records point-in-time test/behavior validation runs (including
  GPU results that can't be reproduced in a CPU-only environment) — treat it as a
  changelog of verification evidence, not living documentation.
- There is an untracked top-level `racesense3d/` directory (a separate nested
  checkout with its own `.venv`) alongside the tracked `src/racesense3d/` package —
  these are different things; don't confuse them, and don't assume the top-level
  one needs to stay in sync with anything.
