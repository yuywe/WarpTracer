# PPO on the elevated oval

PPO controls steering and forward throttle/braking from 108 LiDAR beams and six
vehicle/terrain features. Cars are independent copies of the oval. Training uses
lean physics, uncached EDT sensing and 60 Hz control (four 240 Hz substeps).

## Run it

```bash
uv run --extra rl warptracer train smoke
uv run --extra rl warptracer train oval
uv run --extra rl warptracer evaluate
```

The smoke preset uses CPU and checks the complete training/checkpoint path. It
is too short to learn to race. `oval` requires CUDA and uses 1,024 cars; `large`
uses 4,096. Start with `oval`: higher simulator throughput does not establish
which batch size trains best. PyTorch 2.8 is installed only with the `rl` extra;
existing demos and benchmarks do not need it. Its CUDA dependencies require
several GB of disk space on Linux.

| Preset | Device | Cars | Steps per car per rollout | PPO updates | Aggregate transitions |
| --- | --- | ---: | ---: | ---: | ---: |
| `smoke` | CPU | 8 | 32 | 3 | 768 |
| `oval` | CUDA | 1,024 | 64 | 300 | 19,660,800 |
| `large` | CUDA | 4,096 | 32 | 300 | 39,321,600 |

One PPO update collects a rollout, computes GAE, then performs four shuffled
passes over eight minibatches, stopping early if the epoch's approximate KL
exceeds 0.02. Hidden layers have 128 units (64 in smoke); learning rate is
0.0003, gamma 0.99 and lambda 0.95. Training starts in float32 with ordinary
PyTorch autograd/Adam. Torch compilation and mixed precision are future measured
optimizations, rather than prerequisites to establish a correct baseline.

Training writes `outputs/ppo/training.jsonl`, `config.json` and `checkpoint.pt`.
The checkpoint includes actor, critic, optimizer, RNG state, episode buffers,
vehicle state and update counters. It is saved atomically before training, after
the first update of each run, every 25 updates, and at completion. An interrupted
update preserves the last saved checkpoint, including the initial state if no
update has finished. Resuming in the same output directory removes incomplete
log records and records newer than that checkpoint. Resume adds updates to its
saved configuration:

```bash
uv run --extra rl warptracer train oval --resume outputs/ppo/checkpoint.pt
```

`--updates`, `--envs`, `--seed` and `--output` are optional overrides, not required
for ordinary runs. A resume restores the saved car count and architecture;
changing the count while resuming is rejected. CPU/GPU migration is supported,
but exact stochastic continuation is only expected on the same device/runtime.

## Colab

Open [the PPO notebook](https://colab.research.google.com/github/yuywe/WarpTracer/blob/performance-prototype/notebooks/ppo_colab.ipynb)
and select a GPU runtime. Run setup, then training with `preset = "oval"`.
The notebook streams progress, evaluates the checkpoint, plots training history,
and downloads a zip containing the checkpoint, logs and interactive HTML replay.
The `large` preset is optional after `oval` succeeds. No account or external
logging service is required.

Evaluation creates a separate one-car environment and uses deterministic actions
(the tanh of the actor mean). It stops at the first lap, collision, stall or time
limit. Open `outputs/ppo/evaluation.html`; `evaluation.json` reports return,
lap progress and the end reason. A single replay is a behavior check, not a
multi-seed policy-quality result. Evaluation retains the terminal state instead
of auto-resetting. The replay includes the initial pose, 30 Hz samples and the
final pose/scan, including episodes that end on their first transition. You can
orbit the camera, pause and scrub the recording; there is no live training viewer.

## Environment contract

`RacingEnv.step(actions)` returns borrowed Torch tensors:
`(observation, reward, terminated, truncated)`. These buffers are overwritten
by the next step; rollouts copy them. `final_observation` is the freshly sensed
state before auto-reset. Finished cars receive new random centerline phases and
small heading perturbations; survivors keep their state and scan. RNG sequences
are owned by individual cars and depend on their episode counts. The second
LiDAR launch marches only reset cars, not the whole batch.

CUDA arrays alias Torch buffers without host transfers. Warp kernels and captured
environment graphs run on the current Torch CUDA stream. Policy inference and
losses run on that same device. Graph construction uses a retained nondefault
Warp stream because Torch's default CUDA stream cannot be captured. Setup waits
order the two streams; graph replay follows the caller's current Torch stream.
Nested scopes reuse an existing wrapper for the same external stream handle.
The rollout never calls `done.any()` or copies
per-car observations to CPU. Aggregate episode statistics, timing and LiDAR
convergence checks are inspected between updates. Training refuses an exhausted
EDT scan instead of silently learning from an incomplete one.

| Input/output | Meaning |
| --- | --- |
| Action 0 | Absolute steering request in [-1, 1], scaled to ±0.418 rad; existing steering-rate limit applies |
| Action 1 | Positive throttle, negative brake, zero coasting; throttle is disabled at 6 m/s |
| Observation 0–107 | Ranges clipped/normalized to 10 m; misses are 1 |
| Observation 108–111 | Forward/lateral velocity, yaw rate, applied steering; fixed physical scaling |
| Observation 112–113 | Terrain grade along/across the vehicle heading, normalized to 20% |
| Reward | 5 × signed local centerline arc progress in meters, minus 0.01 per simulated second |
| Collision/off-track/nonfinite/stall | Terminal, reward -5; stall means speed ≤0.05 m/s for 600 transitions |
| Completed forward lap | Terminal, additional reward +5 |
| Episode time limit | Truncated after 3,600 transitions; retains the pre-reset critic bootstrap |

Spawns start at 0.5 m/s. Observations do not expose global coordinates or the
reward's centerline phase. Lap progress wraps signed angle increments around the
oval, so crossing the ±π boundary neither awards a full lap nor loses progress;
driving backward produces negative progress. Contact counts include every
physics substep, so a contact earlier in a fused transition still ends an episode.

The actor samples a Gaussian latent action and applies tanh. Rollouts store the
latent sample and the transformed log probability with a stable tanh Jacobian.
Updates use the same likelihood; no inverse-tanh is required at saturated actions.
Advantages are normalized across the entire rollout. Value loss is clipped,
gradients are limited to norm 0.5, and an optional entropy term is explicitly the
base Gaussian entropy surrogate (default coefficient zero).

GAE uses **two masks**: real terminal states have no value bootstrap, while time
limits bootstrap from their final observation. Both stop the recursive advantage
trace, preventing rewards in a newly spawned episode from leaking backward.

`rollout_transitions_per_second` includes policy inference, Warp physics, both
observation stages, reset handling and per-step critic bootstrapping. Training
throughput also includes GAE, shuffled minibatch optimization and correctness
checks. Both aggregate all cars. Setup/compilation, checkpoint and log writes are
outside these timings. Compare GPU measurements only after the first update and
inspect several updates; the simulator-only scale benchmark is a different workload.

## Reference review

These are useful compact implementations; neither was run or independently
validated for learning quality here. We reviewed source at pinned revisions and
wrote our implementation independently. No source license was present in either
reviewed tree; no project code was copied.

| Reference | Useful patterns | Differences to account for |
| --- | --- | --- |
| [UCI warporacer, e16e447](https://github.com/uci-f1tenth/warporacer/tree/e16e4473cb86e6ae2e1fa1b078e5a1e04b20d7f2) | Warp/Torch aliases, device rollouts, separate actor/critic, shuffled PPO minibatches, gradient/value clipping, KL checks, aggregate episode logging | Combines crash and time-limit `done`; discards both bootstraps. Includes running observation/reward normalization, adaptive learning rate, compilation and autocast. Its final checkpoint saves policy/observation-normalizer state, not optimizer/full episodes. |
| [warporacer3d, 0d8e059](https://github.com/AlistairKeiller/warporacer3d/tree/0d8e059353d17218a69f949a558b14d84d1b305c) | Compact batched GAE, signed continuous route progress, device buffers, controlled log std and global gradient norm | Now Mojo/MAX, with manual backprop/Adam, a shared 64-unit network, Huber value gradients and fixed strided minibatches. GAE also cuts timeout bootstraps. Resume restores weights/iteration while Adam restarts. |

The UCI code is the closer engineering reference for this Warp/PyTorch stack.
The Mojo version is a useful algorithm/layout reference, not a drop-in backend.
Float16, manually written gradients, normalization and batching choices deserve
separate validation before adopting their performance optimizations.

Algorithm references: [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
and [GAE (Schulman et al.)](https://arxiv.org/abs/1506.02438).
Warp interoperability: [v1.17 stream_from_torch](https://nvidia.github.io/warp/v1.17/api_reference/_generated/warp.stream_from_torch.html).

## Limits

This first task is the analytic elevated oval, uncached EDT LiDAR and lean
terrain-following physics. PNG map import, more tracks, domain randomization,
reverse driving, multi-car interactions and calibrated slope/tire dynamics remain
future work. The environment is a Torch vector interface, not a Gymnasium adapter.
CPU correctness/learning checks do not establish native GPU performance or
convergence across seeds. Inspect the learned replay and lap rate before increasing
the training budget or interpreting throughput as racing quality.
