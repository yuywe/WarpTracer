# Validation

Checked on 2026-10-07 UTC with Linux x86-64, Python 3.12.14, Newton 1.6.0,
Warp 1.17.0, and Viser 1.0.26. SciPy is now an explicit dependency for EDT
preprocessing; the locked package versions remain unchanged.

## Behavior and capture checks

Full suite: **113 passed, 37 skipped** (CUDA unavailable).
This includes the 40 Hz recording and disparity bug regression checks.

## Run commands and presets

The installed `warptracer` entry point was exercised with the README's exact
`uv run warptracer demo` and `uv run warptracer benchmark` commands, without
optional dependency flags. The default demo exported a 30-second navigation
replay, 1,801 scans at 60 Hz, and zero wall-contact substeps. The default quick
benchmark completed three trials for all six workload/execution combinations.
These CPU runs verify the workflows, not new GPU performance.

CLI tests check preset selection, explicit overrides, navigation vs. LiDAR
workloads, fusion variants, recording, replay export, and immediate failure when
a GPU preset is requested without CUDA. Both notebooks' code cells compile;
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

### Large batches, controlled moving scans and sampling cache

Optional per-ray caching keeps the four grid samples until that ray leaves a
bilinear patch. Cached/uncached kernels are specialized separately. CPU checks
compare 257 tilted poses, including ground/roof/sky rays, with identical masks
and range agreement within 2e-5 m. The same terrain representation, 2 mm surface
tolerance and 512-iteration budget are retained. Uncached remains the default.
The lean, disparity and EDT modules disable unused adjoint code generation for
model-free rollouts; Newton and the copied mesh sensor package are unchanged.

The `moving_scan` workload replays a shared 4,096-pose tilted centerline loop
with phase offsets per car. Tests require identical sensor poses across mesh,
uncached EDT and cached EDT, motion, loop wraparound and reset. Captured/fused
and eager/unfused executions agree at 40 and 60 Hz. Moving-scan reports have zero
physics rate and null simulated-speed metrics. It is a prescribed trajectory,
not a recorded physical vehicle path.

CPU 4,096-car checks compare selected scan entries against independent cars.
A separate 4,096-car navigation check compares poses, velocities, controls,
ranges and masks at distinct initial phases against independent runs after two
transitions. This is an indexing/behavior check, not sustained GPU throughput or
PPO validation. CUDA variants of these tests remain skipped locally.

A three-trial CPU API graph check exercised `scale` with 1 / 64 cars, 10 seconds
of the 60 Hz schedule and shortened 0.25-real-second warmup. All 12 execution
parity checks passed; all trials had zero wall contacts and zero exhausted rays.
At 64 cars, cached/uncached ratios were 0.938x for moving scans and 0.954x for
navigation: caching did not help this CPU workload. Uncached EDT/mesh ratios
were 1.446x and 1.099x respectively. These are not predictions for a T4.

The `scale` preset tests 256 / 1,024 / 4,096 cars, mesh / uncached EDT / cached
EDT, moving scans and disparity navigation, with 20 seconds per trial, three
trials and default warmup. Navigation starts are spread around the oval so the
batch does not contain thousands of identical driving states. Schema 5 records
sampling variants and separate cache speed ratios. Both notebook code-cell
compilation and Python 3.10 syntax checks passed. GPU throughput and hosted
Colab execution of this new preset remain to be measured. PPO inference,
learning, reward/termination logic and automatic resets are not implemented.

### Historical T4 result: deleted traversal backend

The user's three-trial, 100-second Tesla T4 comparison used 2.5 cm cells, 108
beams at 60 Hz, fused/graph and default warmup. All printed execution parity
checks passed. Median grid/mesh speed ratios were:

| Cars | Fixed scans | Full navigation |
| ---: | ---: | ---: |
| 1 | 0.334x | 0.731x |
| 64 | 0.471x | 0.736x |
| 256 | 0.239x | 0.695x |

That implementation was slower in every tested case. Its common-hit p99 error
was 31.0 mm, with 31 hit/miss differences and a 6.82 m maximum error. These results
motivate the replacement and are not measurements of EDT marching. Use the same
`benchmark grid` preset on Colab to measure the new implementation.
PNG import, arbitrary-map physics, stacked terrain and slope gravity remain
outside this change.

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

## User-supplied CUDA benchmark (historical: unfused)

Source: uploaded `benchmark.json`, reviewed with this documentation update.
Hardware: **NVIDIA GeForce RTX 4060 Laptop GPU**, Windows 11, Python 3.12.12,
Warp 1.17.0, Newton 1.6.0, NumPy 2.5.3. Physics engine: **lean**.

One car; four 240 Hz substeps per transition; 1080 LiDAR beams at 60 Hz;
recording at 30 Hz. Each of five trials advances ten simulated seconds after
two simulated seconds of warmup. These are medians from the supplied report.

| Workload | Eager transitions/s | Graph transitions/s | Graph substeps/s | Graph speedup |
| --- | ---: | ---: | ---: | ---: |
| Physics only | 4,675 | 57,283 | 229,130 | 12.3× |
| Physics + LiDAR | 4,406 | 29,360 | 117,442 | 6.7× |
| Physics + LiDAR + recording | 2,636 | 6,537 | 26,149 | 2.5× |

All three CUDA graph parity checks report passed. Recomputed medians agree with
the report, and all 30 trials have identical final body poses and velocities.
Recording trials retain 301 frames each. This validates the benchmark workload,
not arbitrary vehicle states or all GPU test cases.

With LiDAR and graph replay, ten simulated seconds take **20.44 ms** at the
median, about **489× real time**. Adding host recording raises this to **91.78 ms**
(about 109× real time). That points to host copying/synchronization as a major
cost in the recording path; it is not a kernel-level profile.

Physics-only graph trials span 9.16–15.06 ms; LiDAR graph trials span
17.04–28.10 ms. These short trials show noticeable variation. Use the
[longer headless benchmark](docs/benchmarking.md#longer-timing-samples) before
drawing fine-grained performance conclusions. CUDA event intervals include
stream idle time; they do not isolate GPU kernel time.

This report compares eager and graph execution of lean physics. It contains
no Newton timings, so it cannot establish lean-versus-Newton GPU speedup.
Parallel-environment throughput remains unmeasured.

## Single-car CPU benchmark (historical: 1080 rays)

Both engines use the same scripted circle workload, four 240 Hz physics substeps
per transition, 1080 rays at 60 Hz in sensing cases, and 30 Hz host recording.
Three trials follow two simulated seconds of warmup each. Lean trials advance
60 simulated seconds; Newton trials advance 10. Medians below measure one car,
not aggregate throughput across parallel environments.

Timing includes stepping, sensing, and optional host recording. It excludes
compilation, construction, warmup, reset, validation, HTML export, and disk writes.
Recording retains body/control/scan samples; it does not generate the viewer.

| Engine | Execution | Physics only, substeps/s | With LiDAR, substeps/s | With LiDAR + recording, substeps/s |
| --- | --- | ---: | ---: | ---: |
| lean | eager | 19,632 | 9,068 | 8,429 |
| lean | graph | 1,615,713 | 16,256 | 12,750 |
| newton | eager | 408 | 392 | 401 |
| newton | graph | 5,010 | 3,769 | 3,612 |

Divide substeps/s by four for environment transitions/s.

Reproduce the historical CPU measurements:

```bash
uv run --locked python -m warptracer.benchmark --device cpu --physics lean --backend both --integrator unfused --warmup-wall-seconds 0 --lidar-beams 1080 --seconds 60 --trials 3 --output outputs/lean-cpu.json
uv run --locked python -m warptracer.benchmark --device cpu --physics newton --backend both --integrator unfused --warmup-wall-seconds 0 --lidar-beams 1080 --seconds 10 --trials 3 --output outputs/newton-cpu.json
```

The physics-only lean graph trial is very short, so its rate is especially
sensitive to timer resolution and machine load. CPU results are indicative,
not a GPU speed claim. JSON reports retain individual trials and timing spread.
The historical demo scanned at 30 Hz; current CLI demos and benchmark presets
both default to 60 Hz sensing.

## Driving and playback (historical: 1080 rays)

All three ten-second lean driving demos export Viser HTML, trajectory NPZ, and
metadata JSON. Each has 301 body samples and 301 mounted scans with 1080 beams
at the then-default 30 Hz, with all returns valid. Acceleration/braking peaks near
1.69 m/s and stops; the circle completes a lap; the S-turn changes steering
direction and stops. The scene still uses one box car, one floor, and four walls.

All code cells in both notebooks compile. The benchmark notebook clones or switches to
performance-prototype and uses the same repository-root uv environment as the
driving notebook. Actual hosted Colab and native browser playback have not been
tested here.

## Model limits

Lean uses a planar dynamic bicycle with illustrative parameters, two
friction-limited axle forces and an implicit lateral/yaw prediction. Height is
fixed in the room; the oval constrains height/tilt to its road surface and reports
vertical motion. Chassis-footprint checks stop velocity at room walls or oval
barriers. The force model does not simulate gravity along slopes, suspension,
airborne motion, calibrated tire slip curves, wheel spin, or reverse throttle.
Newton remains available for the earlier room force/contact behavior.

LiDAR is instantaneous and noise-free. Mesh sensing uses the unchanged sensor
package; the optional grid backend approximates the oval with height/occupancy
arrays. The host vehicle is excluded. Mesh sensor geometry matches playback;
the grid approximation can differ at rasterized boundaries.
Independent environments now share static geometry and run in batches.
Moving obstacles, scan distortion, per-environment auto-reset,
rewards, and PPO are not implemented.
