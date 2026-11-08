# Validation

Point-in-time evidence; older entries may mention Newton, removed presets or
removed flags, which existed when they ran.

Checked on 2026-10-20 UTC with Linux x86-64, Python 3.12.14, Newton 1.6.0,
Warp 1.17.0, Pyglet 2.1.19, and ImageIO/FFmpeg. SciPy is an explicit dependency for EDT
preprocessing; PyTorch 2.8.0 is an optional `rl` dependency for training.

## Behavior and capture checks

Full suite: **160 passed, 53 skipped** (CUDA unavailable), including optional
PPO tests and Warp recording/playback checks.
This includes the 40 Hz recording and disparity bug regression checks.

## Warp playback replacement

The active visualization path now uses Warp OpenGL; Viser and its browser/server
dependency stack have been removed. Demos and evaluation save self-contained NPZ
recordings without initializing graphics. Optional `--view` opens native playback;
`--video` renders an MP4 offscreen. Both Colab notebooks use videos. Historical
HTML measurements below describe the older implementation.

CPU checks cover scene/scan round trips for room, oval and PNG maps, old demo
sidecars, XYZW rotations, box half-extents, independent body/scan timestamps,
invalid-return hiding, terminal-frame retention, and pause/seek/restart controls.
An actual MP4 is encoded/decoded using a stub renderer to check frame ordering
and terminal image content; cleanup after a pixel-read failure is checked.
Import checks confirm recording/training do not import Pyglet, warp.render or
Viser. CLI checks exercise both viewing/export dispatch paths without a context.
All three notebooks pass schema validation and Python 3.10 syntax checks.

A real offscreen attempt reached Warp OpenGL on the CPU-only host, but failed in
Warp 1.17's GL-buffer fallback because pinned allocation still needs a CUDA
driver. The application now reports this requirement before creating graphics
and preserves recordings. Native CUDA/OpenGL output, camera interaction, EGL on
Colab and rendering performance remain unverified here; the native GPU smoke test
is included and skipped locally. Rendering and encoding remain outside simulation
and PPO throughput measurements.

## PNG training maps

PNG integration checks use an offset rectangular ring rather than an oval.
They cover image/world Y orientation and scale, opaque white versus gray/alpha
pixels, inverted brightness, automatic/reversed routes, explicit CSV routes,
rejection of ambiguous topology and unsafe clearance, route previews, and
portable bitmap/route serialization. A one-beam scan hits the expected image
wall in world coordinates without exhausting its march budget.

Physics checks cover fused/unfused substep parity, the entire conservative chassis
circle, and a deliberately oversized substep that would otherwise cross a thin
wall between free endpoints. Episode checks cover CPU eager/graph parity,
continuous signed progress across the route wrap, lap termination, selective
resets, and exact next-transition equality after restoring the route cursor.
The CLI completes PPO smoke training, resumes after the source PNG/JSON are
deleted, and exports evaluation JSON/NPZ from embedded checkpoint geometry.

The map preview was visually inspected. All source/notebook code parses with
Python 3.10 syntax. Actual Colab uploads, native CUDA execution of the new map
kernels, and policy convergence/performance on user maps remain unverified here.

## Run commands and presets

The installed `warptracer` entry point was exercised with the README's exact
`uv run warptracer demo` and `uv run warptracer benchmark` commands, without
optional dependency flags. The default demo exported a 30-second navigation
replay, 1,801 scans at 60 Hz, and zero wall-contact substeps. The default quick
benchmark completed three trials for all six workload/execution combinations.
These CPU runs verify the workflows, not new GPU performance.

CLI tests check preset selection, explicit overrides, navigation vs. LiDAR
workloads, fusion variants, recording, replay export, and immediate failure when
a GPU preset is requested without CUDA. All three notebooks' code cells compile;
actual hosted Colab execution remains untested here.

The default scan is now 108 rays; both demo and benchmark accept
`--lidar-beams 1080` for dense scans. Moving scans at both resolutions match
analytic geometry, and benchmark tests verify selected resolution, eager/graph
parity, recorded frame counts, and JSON metadata. A one-second 108-ray circle
replay exported successfully with 31 scans and all returns valid. No new GPU
performance measurement is claimed for 108 rays.

The original Newton driving, free-body, and LiDAR tests still pass. New lean
checks cover acceleration, braking without sustained reverse, fixed height,
normalized orientation, stationary steering, reset, symmetric turns and steering
rate limits, zero-grip coasting under throttle/steering, and rotated chassis
containment with stop-on-wall behavior.

For both engines, captured and eager transitions agree after changing speed and
steering requests and after reset, including body poses, velocities, and mounted
LiDAR. Unsafe odd substep counts and inconsistent sensor schedules are rejected.
The benchmark independently checks eager/graph poses, velocities, applied
controls, scan poses, ranges, masks, and device step counts before timing.

Thirty-seven CUDA tests were skipped in the local CPU-only environment. CPU graph replay
uses Warp's CPU API capture. The user-supplied GPU benchmarks below separately
confirm execution parity for their workloads; they do not establish that the full
CUDA test suite or the new sampling-cache/moving-scan paths pass.
No automatic fallback hides capture errors.

## Disparity navigation

The 2026-10-05 regression checks reproduce and prevent stale clearance after a
corner-target override, a turn that never releases with a 3 m scoring cap, and
manual commands being ignored after a direct controller update. An empty search
sector is rejected. Corner targets now retain the filtered distance of the actual
selected beam; braking uses that clearance as well as the forward corridor.
Turn-release geometry uses sensor ranges independently of the scoring cap.
The comparison with `uci-f1tenth/race_stack` is documented in `docs/navigation.md`;
that implementation uses a sliding-window minimum rather than disparity extension.

CPU tests cover extension in both directions, overlapping disparities, unchanged
raw scans, steering toward the open side, braking for blocked/invalid scans,
approach braking, independent batch commands, controller reset, and parity across
fused/unfused and eager/graph execution. Three differently positioned cars are
compared against independent single-car controllers over ten simulated seconds,
then repeated after reset. Navigation tests at 30, 40 and 60 Hz require movement
and turns for 60 simulated seconds with zero wall-contact substeps.

A 60-second, 60 Hz recorded demo traveled about 51.6 m, peaked at 1.28 m/s, and
finished moving at 0.54 m/s with zero wall-contact substeps. It exported HTML,
NPZ and JSON, with 3,601 scans; its XY trajectory and speed trace were inspected.
The headless navigation benchmark completed with 1 and 64 independent cars on
CPU and passed reference validation. After the fixes, all eager/graph and
fused/unfused combinations were checked again with those batch sizes over ten
simulated seconds; the 60-second recorded demo again traveled 51.6 m with zero
wall-contact substeps. These CPU timing runs used one trial and a shortened
0.25-second real warmup; they are behavior/parity checks, not GPU performance data.
The user's later mesh-oval CUDA report below separately covers disparity
throughput and reference parity; the new grid backend remains untested on CUDA.

The unmodified published simulation previously achieved 5.38 million aggregate
transitions/s on the user's Tesla T4 with 256 cars and 108-ray LiDAR at 60 Hz.
That report used scripted commands; it is not a disparity-controller measurement.

## Elevated oval

`uv run warptracer demo oval` exported the 120-second replay with 7,201 scans,
all valid. The car traveled about 130.5 m and completed 4.03 laps of angular
progress, with zero barrier-contact substeps and a 0.400 m height range. Peak
speed was 1.28 m/s; final speed was 1.23 m/s. The shared track geometry, recorded
path and elevation trace were inspected. This was a CPU workflow/behavior check,
not a GPU throughput measurement.

Tests require multiple complete laps and zero contacts at 30, 40 and 60 Hz.
They compare vehicle height/normal against the analytic surface, check mounted
LiDAR tilt and world vertical velocity, and raycast the sampled road to within
1 mm of its analytic height. Barrier meshes have closed seams and nondegenerate
triangles. Outer and inner impacts keep the full chassis inside the lane, stop
velocity, and clear contact counters on reset.

Three distinct tilted/tangent spawns are compared against an unfused/eager
reference across all four CPU execution combinations, including scans, controls,
goals and contacts. Reset preserves their initial headings. The short demo and
oval benchmark preset tests verify output/track metadata and that benchmark
reports do not overwrite demo metadata. CUDA oval tests are skipped locally;
the user's subsequent Colab report is recorded below.
The ten-second CPU oval benchmark also passed full-state reference validation
with 1 and 64 cars, using one trial and a shortened 0.25-second real warmup.

The user supplied a Tesla T4 mesh-oval report on 2026-10-07 UTC: three trials,
100 simulated seconds per car, graph/fused, 108 rays at 60 Hz, and the default
warmup. Median aggregate navigation transitions/s were 6,490 / 372,202 / 919,642
for 1 / 64 / 256 cars; corresponding batch rates were 6,490 / 5,816 / 3,592.
Trial variation was under 2% and all printed unfused/eager parity checks passed.
This pasted report validates that workload; it does not establish grid results
or execution of every CUDA test.

## Trimmed codebase

Checked on 2026-11-05, same machine as the entries below, after removing Newton and
the scan-only benchmark workloads and merging the two bicycle kernels. Full suite:
**247 passed, 1 skipped**. 120-second disparity demos on the oval and bridge matched
recordings made before the refactor exactly: the maximum difference in poses,
velocities, controls and LiDAR ranges was 0. Retraining the bridge for 300 updates
(seed 0) reproduced the pre-refactor run's totals over the last 100 updates and its
evaluation return of 1364.934. All six benchmark presets ran on cuda:0 with their
parity checks passing, in 2-second single trials that are not throughput comparisons.
Tests in `test_driving.py` and `test_lidar.py` that had defaulted to Newton now run
on lean physics.

## Multi-lap PPO episodes

Checked on 2026-11-05, same machine as the slope gravity run below. Laps now pay
their bonus without ending the episode. Full suite: **267 passed, 1 skipped**.
`uv run warptracer train oval` for 300 updates (seed 0), last 100 updates, with
one deterministic evaluation of the final checkpoint:

| Track | Episodes ended | Laps | Mean laps/episode | Mean length | Evaluation laps in 60 s | Lap time after the first |
| --- | --- | --- | --- | --- | --- | --- |
| Bridge | 1,984 | 9,047 | 5.56 | 3,432 of 3,600 | 5 | about 10.0 s |
| Oval | 2,100 | 12,767 | 6.53 | 3,076 of 3,600 | 7 | about 7.5 s |

Neither evaluation crashed; both ended at the time limit. A bridge policy trained
with lap termination also drove 5 laps in 60 s under the new evaluator, at about
10.25 s per lap. Its first lap took 11.95 s, against 11.85 s for the new policy.
This is one seed and one replay per track; the native viewer was not opened.

## Slope gravity

Checked on 2026-11-05 on Windows 11 with an RTX 4060 Laptop GPU (driver 610.88,
CUDA Toolkit 12.9), Warp 1.17.0. Full suite: **261 passed, 1 skipped** (the CLI
check that needs a host without CUDA); before the change it was 231 passed, 1
skipped. `test_slope_gravity.py` checks coasting against the analytic plane result,
round trips, brake hold, cross slopes, flat-ground exactness and fused/graph parity
on the oval and bridge. `uv run warptracer benchmark oval --seconds 5 --trials 1`
passed unfused/eager parity for 1, 64 and 256 cars; with one trial, its rates are
not a throughput comparison.

Disparity demos on cuda:0, 120 simulated seconds, before and after the change:

| Track | Laps | Distance | Climbing mean speed | Descending mean speed | Wall contacts |
| --- | --- | --- | --- | --- | --- |
| Oval, before | 4 | 130.4 m | 1.20 m/s | 1.21 m/s | 0 |
| Oval, after | 3 | 128.3 m | 1.00 m/s | 1.41 m/s | 0 |
| Bridge, before | 3 | 129.9 m | 1.01 m/s | 1.02 m/s | 0 |
| Bridge, after | 2 | 124.7 m | 0.71 m/s | 1.33 m/s | 0 |

Lap counts are whole laps, so small distance changes can cross a lap boundary.
`uv run warptracer train oval --bridge` (300 updates, seed 0) completed a lap in
4,572 of 4,713 episodes (97.0%) over the last 50 updates. One greedy evaluation
finished a lap in 11.95 s. This is one seed and one replay; the native viewer was
not opened.

A wrapped-launch audit found every kernel on cuda:0 and no host readbacks while
stepping lean room/oval/bridge, Newton room and PPO; `test_device_placement.py`
also covers PNG maps. Graph replays run captured kernels, so they were audited only
by state-array device and absence of readbacks.

## Layered tracks (bridge)

Checked on 2026-10-29 on Windows 11 with an RTX 4060 Laptop GPU (driver 610.88,
CUDA Toolkit 12.9), Warp 1.17.0. Full suite: **226 passed, 1 skipped** (the CLI
check that needs a host without CUDA). `test_layers.py` runs on both CPU and
`cuda:0`, so the graph/unfused, eager/fused and graph/fused layered variants
matched unfused eager on the GPU, as did the under-then-over disparity drive.

`uv run warptracer demo bridge` ran 120 simulated seconds on cuda:0 (graph/fused)
in 6.19 s, 4,652 physics steps/s including pose recording. It completed 3 laps
with zero wall-contact substeps; 7,201 scans of 108 beams were all valid. Peak
speed was 1.37 m/s. Recorded height spans 0.103-0.803 m, and samples within
0.5 m of the crossing sit at both about 0.1 m and 0.8 m, so the car passed both
under and over the 0.70 m deck. This is one default-geometry run, not a
throughput benchmark; the native viewer was not opened.

PPO on the bridge (same machine and day): the suite grew to **231 passed, 1
skipped** with five layered PPO tests (route-layer respawns on CPU and CUDA,
eager/graph parity, route cursor staying on the deck over the crossing, exact
resume, smoke train + evaluate). `uv run warptracer train oval --bridge` (seed 0,
1,024 cars, 300 updates) took 281 s of rollout+update time; rollout ran at
roughly 120k-150k transitions/s, with some updates near 50k/s on the shared
laptop GPU. The first successful laps appeared at update 21; over the last 50
updates, 4,837 of 4,849 finished episodes were laps (99.8%). Deterministic
evaluation completed a lap in 11.35 s (681 transitions, return 234.9, peak
4.8 m/s), passing under and over the deck. One seed and one evaluation episode,
not a convergence study; the replay was not inspected in the viewer.

## EDT distance-field LiDAR

The old `heightfield.py` cell/tile traversal has been deleted. The `grid` backend
now uses `distance_field.py`: one-time SciPy EDT preprocessing, normalized
bilinear signed-clearance fields, and a Warp 3D ray marcher. Terrain-gradient
bounds guide floor/roof jumps, and walls retain finite height. Mesh sensing is
still the default and `src/racesense3d` remains unchanged.

Independent tests cover ground from above/below, holes, sky and outside-domain
origins, bilinear quadratic terrain intersections, range factors, near clipping,
wall entry/inside exit, roofs and passing above walls. Random 3D rays are compared
with an independent sloped-plane/rectangular-obstacle mesh. Away from rounded
EDT corner contours they agree within 1 cm; the corner exception is explicitly
measured rather than claiming exact rectangular geometry. EDT sign, zero-contour
placement and bilinear gradient bounds are checked. A deliberately insufficient
iteration budget is reported separately from ordinary misses, and benchmarks
reject exhausted scans. Reset clears statistics without rebuilding the fields.

The new default comparison covers 432 identical poses and 46,656 rays: 44,525
common hits, 5.54 mm median error, 15.4 mm p95 and 27.7 mm p99. There are 29
hit/miss disagreements (0.062%), 12 common-hit errors above 1 m and a 6.86 m
maximum error. Raster/EDT corner and grazing decisions still cause large first-hit
changes. Worst-ray origins, directions and both ranges remain in the report.
Maximum iterations observed were 132 of 512, with zero budget-exhaustion events.
See [the guide](docs/grid-lidar.md) for how to interpret tolerance and errors.

The exact `uv run warptracer demo oval-grid` command exported HTML/NPZ/JSON:
four completed laps, 7,201 valid scans, a 0.400 m elevation range and zero
contacts. Peak speed was 1.28 m/s and final speed was 1.22 m/s. March convergence
counts are included in the JSON. CPU elapsed time was 1.024 s including recording;
this verifies the workflow and does not predict GPU speed.

Three distinct spawns pass unfused/eager reference checks for graph and fused
variants, including scans, controller state, contacts and reset. At 40 Hz, batch
entries match independent single cars. The scan-only benchmark keeps sensor poses
identical across backends, never advances physics, and reports zero physics rate
and null simulated-speed metrics. Replay metadata and schema-4 reports identify
the EDT algorithm and separate convergence counts from timing. Both notebooks'
code cells compile. A CUDA driver is unavailable locally. The user's subsequent T4 result below
validates the original EDT workload; it does not validate the newer cache or
moving-scan variants.

A three-trial CPU check used 1 and 64 cars, 10 seconds of the 60 Hz schedule,
fused CPU API graph replay, and a shortened 0.25-real-second warmup. All eight
execution parity checks passed, every trial reported zero contacts, and EDT
reported zero exhausted rays. Grid/mesh median speed ratios were 0.909x / 1.428x
for fixed scans and 1.047x / 1.017x for navigation (1 / 64 cars). Single-car samples
were short and these results are not a native CUDA speed prediction. The hosted
Colab setup and GPU benchmark must still be run on the T4.

### User-supplied T4 result: original EDT marcher

The user supplied a three-trial Tesla T4 report with 100 seconds per car,
graph/fused, 108 rays at 60 Hz and default warmup. All 12 execution-parity checks
passed. The identical-pose comparison matched the CPU metrics at printed
precision: p95 15.4 mm, p99 27.7 mm, maximum 6.8555 m, 29 / 46,656 validity
disagreements, maximum 132 march iterations and zero budget exhaustion.
Completion also implies timed EDT trials passed their exhaustion checks.
The pasted output does not include wall-contact counters or lap progress.

| Cars | Mesh scans/s | EDT scans/s | Mesh navigation transitions/s | EDT navigation transitions/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16,946 | 13,250 | 6,480 | 7,071 |
| 64 | 692,931 | 673,781 | 359,106 | 385,929 |
| 256 | 1,230,670 | 1,573,477 | 885,550 | 1,028,179 |

Rates aggregate all cars. Grid/mesh scan ratios were 0.782x / 0.972x / 1.279x;
navigation ratios were 1.091x / 1.075x / 1.161x. Fixed scans have identical poses;
closed-loop navigation paths can differ. These measurements precede sample
caching, distributed navigation starts and the new moving-scan workload.

### Scale grazing-ray budget correction

The user's T4 `scale` run passed all 18 execution-parity checks but aborted
in the first 4,096-car cached-EDT navigation trial: two ray casts exhausted
512 iterations. The reported identical-pose comparison had no exhaustion;
it did not cover the later navigation poses. The completed moving-scan
first trial measured 1,887,754 aggregate transitions/s for mesh, 2,710,511
for cached EDT and 3,040,581 for uncached EDT. These are single-trial results,
not three-trial medians or PPO training measurements.

CPU reproduction found exhausted rays at cars 1522 and 1526, beam 64, in
the first 50 navigation transitions. One isolated shallow ray at transition
39 converged after 587 steps to a valid hit near 5.347 m with an increased
budget. The default budget is now 2,048; geometry, tolerance and conservative
jump bounds are unchanged. Rays still exit as soon as they converge, and
benchmark rejection of exhausted casts remains enabled. Regression tests
require this ray to exhaust the historical 512-step budget and converge with
the new default, for both cached and uncached sampling. CUDA versions remain
skipped locally pending a hosted rerun. The full suite passed 115 tests with
39 CUDA skips. Both reproduced cars completed 1,200 navigation transitions
(20 simulated seconds) with cached and uncached sampling, identical final
poses, a maximum of 695 iterations and no exhausted casts. The full
4,096-car uncached-EDT navigation workload also completed all 1,200 transitions
on CPU with a maximum of 695 iterations and zero exhausted casts. This is a
convergence check, not a GPU performance result.

## PPO and batched episodes

The independent PPO implementation references UCI warporacer at e16e447 and
warporacer3d at 0d8e059; source review and differences are documented in
`docs/ppo.md`. The references were not executed or audited for policy quality.
The environment exposes borrowed Torch views of Warp observations and episode
outputs, captures physics/sensing/reward/reset operations, and uses the current
Torch CUDA stream. Selected reset scans preserve surviving cars' ranges and
iteration statistics. Policy actions use tanh-transformed Gaussian likelihoods.

Tests independently check transformed likelihood against Torch's distribution
transform, saturated-action finiteness, positive/negative surrogate clipping,
hand-computed time-limit and terminal GAE, copied rollout observations, finite
parameter changes, signed angle wrap and lap termination, collision precedence,
nonfinite-state recovery, selective timeouts, eager/captured parity, and 4,096-car
mask/buffer indexing. Resume reproduces the next CPU update exactly, including
policy weights, vehicle poses, reward/KL metrics, optimizer and RNG state.
Native CUDA/stream tests are included but skipped locally without a driver.

A follow-up review reproduced and fixed three recovery/recording gaps: no replay
for a one-transition episode, no checkpoint if the first update was interrupted,
and stale or incomplete JSONL records when resuming an older saved update. New
regressions verify initial and terminal playback samples, eager/captured terminal
pose and scan retention without respawn, an initial recovery checkpoint, and log
rewinding to the resumed checkpoint. PPO rejects evaluation recording mode;
existing checkpoints without the new auto-reset field still resume exactly.
All three notebooks validate against the notebook schema without warnings and
their code cells pass Python 3.10 syntax checks. Browser interaction and native
GPU execution remain untested locally.

The first hosted T4 PPO attempt failed before training during graph capture with
`Warp error: unknown stream`. Warp v1.17 source and its PyTorch interoperability
guidance identify the null/default Torch CUDA stream as unsuitable for capture.
Graph construction now uses a retained Warp-owned nondefault stream with explicit
setup dependencies. Replay remains on the caller's current Torch stream. Nested
scopes reuse matching stream wrappers to avoid unregistering a still-used handle.
Three CPU stream-contract simulations cover default/nondefault caller handles,
setup ordering and nested wrapper reuse; these are not native GPU execution.
Native PPO tests cover both Torch default and nondefault streams but remain
skipped locally without a CUDA driver; the hosted retry is still required.

A CPU learning check used seed 7, 64 cars, 64 rollout steps, 64-unit hidden layers,
four epochs/eight minibatches, and 200 updates (819,200 aggregate transitions).
Before and after training, a separate seed-99 evaluation ran 32 cars for 1,200
transitions (20 simulated seconds per car), with deterministic actions. The
untrained policy had mean step reward 0.0100, 15 completed failure episodes and
zero completed laps. The trained policy had mean step reward 0.3148 and 64
completed episodes, all successful laps. Both evaluations had zero exhausted
LiDAR rays; the trained evaluation's maximum was 721 march iterations.
This is one training seed on the analytic oval, not broad convergence evidence.

The saved policy also passed the actual `warptracer evaluate` workflow: one
successful lap, return 181.015, 560 transitions (9.33 simulated seconds), and
zero exhausted rays. Viser HTML, trajectory NPZ and metrics JSON were exported.
No native GPU training or interactive browser/hosted Colab run was performed
locally. CPU rates do not predict T4 training throughput. The smoke training
entry point, checkpoint save/resume and all notebook code cells are checked.

## Fusion, batches, and wall-clock warmup

The shared bicycle step was compared against the pre-refactor kernel from commit
`f0d9f02`: all saved poses and velocities in the ten-second accelerate/brake,
circle, and S-turn CPU trajectories agree within 1e-6 absolute/relative tolerance.

New CPU tests compare every entry of three-car batches against separate single-car
runs, using different commands, acceleration, braking, turns, wall collisions,
reset, and mounted LiDAR. All eager/graph and fused/unfused combinations pass at
four and six substeps (60 and 40 Hz sensing). A 257-car test checks indexing beyond
one thread block and compares selected cars against individual runs.
Warmup tests check the elapsed wall-clock minimum and restoration of time-zero
state/scans. Benchmark tests check batch counts, aggregate throughput units,
40 Hz headless operation, resolution selection, and the report schema.

The default benchmark now requires two real seconds of warmup before every trial.
Historical reports below used only two simulated seconds. A variation flag marks
max/min elapsed time above 1.2; this does not diagnose the cause of variation.

An exploratory CPU API graph sweep ran three trials, ten simulated seconds each,
with 108 rays at 60 Hz and a shortened **0.25-real-second warmup**:

| Cars | Integrator | Physics aggregate transitions/s | With LiDAR aggregate transitions/s |
| ---: | --- | ---: | ---: |
| 1 | unfused | 498,108 | 38,775 |
| 1 | fused | 696,182 | 40,861 |
| 64 | unfused | 1,861,329 | 44,883 |
| 64 | fused | 2,062,047 | 44,047 |

These are CPU measurements, not predictions for the user's GPU. Physics-only
single-car samples last about one millisecond; their rates are especially
sensitive to overhead. Fusion has little effect on the CPU LiDAR batch workload.
All eight configurations passed full-array comparison against unfused eager
execution before timing. The 64-car fused physics timing had max/min > 1.2.

Reproduce this CPU sweep:

```bash
uv run --locked python -m warptracer.benchmark --device cpu --backend graph --integrator both --envs 1 64 --cases physics lidar --seconds 10 --trials 3 --warmup-wall-seconds .25 --output outputs/fusion-batch-cpu.json
```

The README and Colab benchmark provide GPU runs with the default two-real-second
warmup. New fused/batched native CUDA performance remains unmeasured here.

## Model limits

Lean uses a planar dynamic bicycle with illustrative parameters, two
friction-limited axle forces and an implicit lateral/yaw prediction. Height is
fixed in the room; the oval and layered tracks constrain height/tilt to the road,
report vertical motion and apply gravity along the slope. Chassis-footprint checks
stop velocity at walls. The force model does not simulate suspension, airborne
motion, calibrated tire slip curves, wheel spin, or reverse throttle.

LiDAR is instantaneous and noise-free. Mesh sensing uses the unchanged sensor
package; the optional grid backend approximates the oval with height/occupancy
arrays. The host vehicle is excluded. Mesh sensor geometry matches playback;
the grid approximation can differ at rasterized boundaries.
Independent environments now share static geometry and run in batches.
Moving obstacles and scan distortion are not implemented. The PPO
interface adds rewards, episode termination and individual automatic resets; it does not add car-to-car interactions or calibrated
vehicle/terrain dynamics.
