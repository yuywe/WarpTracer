# Validation

Checked on 2026-10-02 with Linux x86-64, Python 3.12.14, Newton 1.6.0,
Warp 1.17.0, and Viser 1.0.26, using the unchanged root uv.lock.

## Behavior and capture checks

`uv run --locked --extra viz --extra dev pytest -q`: **32 passed, 5 skipped**.

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

Five CUDA tests were skipped in the local CPU-only environment. CPU graph replay
uses Warp's CPU API capture. The user-supplied GPU benchmark below separately
confirms lean CUDA graph/eager parity for its workload; it does not establish
that the full CUDA test suite or the Newton CUDA backend passes.
No automatic fallback hides capture errors.

## User-supplied CUDA benchmark

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

## Single-car CPU benchmark

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

Reproduce the CPU measurements:

```bash
uv run --locked python -m warptracer.benchmark --device cpu --physics lean --backend both --seconds 60 --trials 3 --output outputs/lean-cpu.json
uv run --locked python -m warptracer.benchmark --device cpu --physics newton --backend both --seconds 10 --trials 3 --output outputs/newton-cpu.json
```

The physics-only lean graph trial is very short, so its rate is especially
sensitive to timer resolution and machine load. CPU results are indicative,
not a GPU speed claim. JSON reports retain individual trials and timing spread.
The original demo scans at 30 Hz; benchmark sensing is twice that rate.

## Driving and playback

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
No moving obstacles, scan distortion, navigation controller, parallel
environments, or PPO are added in this milestone.
