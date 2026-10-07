# Validation

Checked on 2026-10-07 UTC with Linux x86-64, Python 3.12.14, Newton 1.6.0,
Warp 1.17.0, and Viser 1.0.26. Dependency versions are unchanged; uv.lock now
includes Viser in the base installation as well as the compatibility viz extra.

## Behavior and capture checks

Full suite: **107 passed, 32 skipped** (CUDA unavailable).
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

Thirty-two CUDA tests were skipped in the local CPU-only environment. CPU graph replay
uses Warp's CPU API capture. The user-supplied GPU benchmark below separately
confirms lean CUDA graph/eager parity for its workload; it does not establish
that the full CUDA test suite, the Newton CUDA backend, or the new fused/batched
CUDA paths pass.
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

## Height-field LiDAR experiment

Grid sensing is opt-in (`demo oval-grid` or `LidarConfig(backend="grid")`). It
uses bilinear terrain and extruded occupancy cells, with conservative 8x8 tile
bounds and full 3D rays. No sensor mesh/BVH is constructed. The sensor package
in `src/racesense3d` remains unchanged as the reference.

Independent checks cover sloped ground from above/below, holes, vertical and
horizontal rays, sky misses, outside-grid origins, exact cell boundaries,
partial tiles, quadratic bilinear intersections, range factors, near clipping,
barrier roofs/sides and rays above barriers. Internal occupied-cell boundaries
are not surfaces. Random rays against a sloped plane and extruded rectangle
agree with independent mesh geometry; random terrain rays agree between tiled
and unaccelerated traversal.

Grid navigation completes multiple laps over 120 seconds with zero contacts,
valid scans and the oval's full elevation range. Three distinct spawns pass
unfused/eager reference checks for graph and fused variants, including controller
state, scans, contact counts and reset. At 40 Hz, batch entries match independent
single-car runs. The fixed-scan benchmark keeps both geometry backends' poses
identical and never advances physics. Tests check replay export, sensor metadata,
comparison errors and speed-ratio fields, and zero physics-rate/null simulation
speed for that scan-only case. CUDA grid tests remain skipped locally.

The exact `uv run warptracer demo oval-grid` command exported HTML/NPZ/JSON,
with four completed laps, 7,201 valid scans, a 0.400 m elevation range and zero
contacts. It finished moving at 1.22 m/s; peak speed was 1.28 m/s. CPU elapsed
time was 1.151 s including recording, which is a workflow check rather than a
GPU prediction.

The 2.5 cm grid's identical-pose comparison samples 432 poses and 46,656 rays:
44,524 common hits, 5.15 mm median error, 17.1 mm p95 and 31.0 mm p99. There are
31 hit/miss disagreements (0.066%), 16 common-hit errors above 1 m, and a 6.82 m
maximum error. These outliers occur around visibility changes at rasterized
boundaries; they must not be hidden behind percentile statistics. The report
retains worst-ray origins, directions and both ranges. See [the grid guide](docs/grid-lidar.md).

`benchmark grid` compares fixed-pose sensing and complete navigation separately,
with backend-specific execution parity checks. A short one-trial CPU run verified
all comparison/report paths for 1 and 3 cars. Its millisecond timing samples do
not establish a speedup or predict the T4. The updated notebooks' code cells
compile; hosted Colab execution of this new preset remains untested here.
Mesh sensing stays the default. PNG import, arbitrary-map physics and slope
gravity are not part of this change.

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
