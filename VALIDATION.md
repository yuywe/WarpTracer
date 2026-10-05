# Validation

Checked on 2026-10-05 with Linux x86-64, Python 3.12.14, Newton 1.6.0,
Warp 1.17.0, and Viser 1.0.26, using the unchanged root uv.lock.

## Behavior and capture checks

Full suite: **69 passed, 23 skipped** (CUDA unavailable).
This includes the 40 Hz recording and disparity bug regression checks.

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

Twenty-three CUDA tests were skipped in the local CPU-only environment. CPU graph replay
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
New controller CUDA execution and its
throughput still need to be checked on a GPU; CPU results do not establish those.

The unmodified published simulation previously achieved 5.38 million aggregate
transitions/s on the user's Tesla T4 with 256 cars and 108-ray LiDAR at 60 Hz.
That report used scripted commands; it is not a disparity-controller measurement.

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
The original demo scans at 30 Hz; benchmark sensing is twice that rate.

## Driving and playback (historical: 1080 rays)

All three ten-second lean driving demos export Viser HTML, trajectory NPZ, and
metadata JSON. Each has 301 body samples and 301 mounted scans with 1080 beams
at the default 30 Hz, with all returns valid. Acceleration/braking peaks near
1.69 m/s and stops; the circle completes a lap; the S-turn changes steering
direction and stops. The scene still uses one box car, one floor, and four walls.

All code cells in both notebooks compile. The benchmark notebook clones or switches to
performance-prototype and uses the same repository-root uv environment as the
driving notebook. Actual hosted Colab and native browser playback have not been
tested here.

## Model limits

Lean is a flat-ground dynamic bicycle with illustrative parameters, two
friction-limited axle forces, an implicit lateral/yaw prediction, and fixed
height. Its rectangular wall check clamps the rotated chassis footprint and
stops velocity. It does not simulate suspension, roll/pitch, free fall,
calibrated tire slip curves, wheel spin, reverse throttle, or terrain following.
Newton remains available for the earlier force/contact behavior.

LiDAR is instantaneous and noise-free, using the unchanged sensor package.
The host vehicle is excluded; static wall/floor geometry is shared with playback.
Independent environments now share static geometry and run in batches.
Moving obstacles, scan distortion, per-environment auto-reset,
rewards, and PPO are not implemented.
