# Benchmarking and graph execution

[Back to the quick start](../README.md) · [Measured results](../VALIDATION.md)

Run on your local GPU or in the benchmark Colab notebook:

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --physics lean --backend both --output outputs/lean-benchmark.json
uv run --locked python -m warptracer.benchmark --device cuda:0 --physics newton --backend both --output outputs/newton-benchmark.json
```

Each command compares physics only, physics plus LiDAR, and physics plus LiDAR
and host recording. The benchmark uses **one car**, four 240 Hz substeps per
transition, **1080 rays at 60 Hz**, and host recording at 30 Hz. The demo uses
30 Hz LiDAR, so its timing is a different workload. Compare batched simulators using aggregate environment transitions/s, the
same sensor workload, and the number of environments. Physics substeps/s
is a different unit.

Five trials per case follow two simulated seconds of warmup. Timing excludes
compilation, construction, reset, parity validation, HTML export, and disk writes.
Recording includes host copies and retained samples, but no HTML export.
The report contains all trials, medians, hardware/software, final states,
physics substeps/s, and environment transitions/s. CUDA event intervals include
stream idle gaps and are not summed kernel durations. `--profile` saves a separate
Python call profile; it does not profile CUDA kernels.

Graph execution is checked against eager state, controls, and scans before timing.
Capture warms the used kernels and solver allocations first. Manual inputs remain
in device buffers and can change between transitions:

```python
from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation

sim = Simulation(scenario="drive", engine="lean", device="cuda:0",
                 lidar=LidarConfig(frequency=60))
runner = TransitionRunner(sim, backend="graph", substeps=4)
sim.set_target_speed(1, steering=.2)
for _ in range(60):
    runner.advance()
runner.reset()
```

Use the runner exclusively for stepping and reset while it owns the simulation.
An even substep count restores the captured input/output buffer identities;
when LiDAR is enabled its rate must equal the transition rate. A device-side
scripted circle controller is used for benchmarking, avoiding host input updates
during the timed loop. Ordinary Python controllers can still set inputs between
transitions.

Without a GPU, use `--device cpu --backend both`. Warp's CPU API capture can test
replay correctness and reduce Python overhead, but its timing is not a CUDA
measurement. A user-supplied RTX 4060 Laptop GPU report confirms lean CUDA graph/eager
parity for the benchmark workload; see the measured results above. Automatic backend selection uses both on CUDA and eager
on CPU.


## Longer timing samples

The default ten simulated seconds can complete in only a few milliseconds on
a GPU. For a more stable headless measurement, increase the simulated duration:

```bash
uv run --locked python -m warptracer.benchmark --device cuda:0 --physics lean --backend both --cases physics lidar --seconds 1000 --trials 5 --output outputs/lean-long.json
```

The default recording case is useful for measuring CPU-copy overhead. Keep it
separate from long headless trials because it retains samples in host memory.
A speedup from eager execution to graphs measures execution overhead reduction
within one physics model; comparing lean against Newton requires a separate
Newton report on the same hardware.
