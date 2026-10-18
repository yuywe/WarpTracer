"""Short PPO presets, resumable checkpoints and deterministic policy replay."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .ppo import PPO, PPOConfig, ActorCritic
from .rl_env import RacingEnv, EpisodeConfig
from .terrain import OvalTrack
from .maps import MapTrack, load_map, restore_track, track_state


PRESETS = {
    "smoke": dict(device="cpu", updates=3, envs=8, rollout=32, hidden=64),
    "oval": dict(device="cuda:0", updates=300, envs=1024, rollout=64, hidden=128),
    "large": dict(device="cuda:0", updates=300, envs=4096, rollout=32, hidden=128),
}


def save_checkpoint(trainer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(trainer.state_dict(), temporary)
    temporary.replace(path)


def _rewind_log(path, iteration):
    """Keep only complete records covered by the checkpoint being resumed."""
    if not path.exists():
        return
    original = path.read_text()
    kept = []
    for line in original.splitlines():
        try:
            record = json.loads(line)
            if record["iteration"] > iteration:
                break
        except (json.JSONDecodeError, KeyError, TypeError):
            break
        kept.append(line + "\n")
    restored = "".join(kept)
    if restored != original:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(restored)
        temporary.replace(path)
        print(f"Restored training log through saved update {iteration}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PPO on an oval or PNG track; presets choose training settings")
    parser.add_argument("preset", choices=PRESETS, nargs="?", default="oval")
    parser.add_argument("--device", help="CPU or CUDA device override")
    parser.add_argument("--updates", type=int, help="Additional PPO updates for this run")
    parser.add_argument("--envs", type=int, help="Override batch size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--map", type=Path, help="PNG map JSON, or PNG with a same-stem JSON sidecar")
    args = parser.parse_args(argv)
    chosen = PRESETS[args.preset]
    device = args.device or chosen["device"]
    updates = chosen["updates"] if args.updates is None else args.updates
    envs = chosen["envs"] if args.envs is None else args.envs
    if updates < 1 or envs < 1:
        parser.error("updates and envs must be positive")
    wp.init()
    if device.startswith("cuda") and (not wp.is_cuda_available() or not torch.cuda.is_available()):
        parser.error("CUDA is unavailable. Select a Colab GPU runtime, or run the smoke preset on CPU")
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    episode = EpisodeConfig(num_envs=envs, seed=args.seed)
    config = PPOConfig(rollout_steps=chosen["rollout"], hidden=chosen["hidden"])
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        episode = EpisodeConfig(**state["environment"]["config"])
        config = PPOConfig(**state["ppo_config"])
        if args.envs is not None and args.envs != episode.num_envs:
            parser.error("Resuming requires the checkpoint environment count")
    try:
        track = restore_track(state["environment"]["track"]) if state else (load_map(args.map) if args.map else OvalTrack())
        if state and args.map and track_state(load_map(args.map)) != state["environment"]["track"]:
            parser.error("Resume uses the saved map; --map must match that checkpoint")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    env = RacingEnv(episode, device=device, track=track)
    trainer = PPO(env, config)
    if state:
        trainer.load_state_dict(state)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.pt"
    settings = {"preset": args.preset, "device": device, "episode": asdict(episode), "ppo": asdict(config),
                "map": track.name if isinstance(track, MapTrack) else "oval",
                "map_source": str(args.map) if args.map else None,
                "additional_updates": updates, "starting_iteration": trainer.iteration,
                "resumed_from": str(args.resume) if args.resume else None}
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    log_path = args.output / "training.jsonl"
    if args.resume and args.resume.resolve() == checkpoint.resolve():
        _rewind_log(log_path, trainer.iteration)
    print(f"PPO {args.preset}: {episode.num_envs} independent cars, {config.rollout_steps} rollout steps, "
          f"{updates} updates on {device}; map: {settings['map']}; checkpoint: {checkpoint}", flush=True)
    mode = "a" if state else "w"
    # There is a valid recovery point even if the first update is interrupted.
    save_checkpoint(trainer, checkpoint)
    last_saved_iteration = trainer.iteration
    try:
        with log_path.open(mode) as log_file:
            for run_update in range(updates):
                metrics = trainer.iterate()
                log_file.write(json.dumps(metrics, allow_nan=False) + "\n")
                log_file.flush()
                print(f"update {metrics['iteration']}: reward={metrics['mean_step_reward']:.4f}; "
                      f"episodes={metrics['completed_episodes']}; laps={metrics['successful_laps']}; "
                      f"KL={metrics['approx_kl']:.4f}; "
                      f"rollout={metrics['rollout_transitions_per_second']:.0f}/s; "
                      f"training={metrics['training_transitions_per_second']:.0f}/s", flush=True)
                # Save completed updates only. An interrupted update never replaces
                # the last valid checkpoint with partially collected/trained state.
                if run_update == 0 or trainer.iteration % 25 == 0:
                    save_checkpoint(trainer, checkpoint)
                    last_saved_iteration = trainer.iteration
            save_checkpoint(trainer, checkpoint)
            last_saved_iteration = trainer.iteration
    except KeyboardInterrupt:
        print(f"Interrupted; checkpoint at {checkpoint} contains saved update {last_saved_iteration}", flush=True)
        return
    print(f"Saved {checkpoint}; evaluate with: uv run --extra rl warptracer evaluate {checkpoint}", flush=True)


def evaluate(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic single-car policy evaluation and interactive replay")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=Path("outputs/ppo/checkpoint.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo/evaluation"))
    args = parser.parse_args(argv)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        parser.error("Unsupported checkpoint version")
    episode = replace(EpisodeConfig(**state["environment"]["config"]), num_envs=1)
    env = RacingEnv(episode, device=args.device, track=restore_track(state["environment"]["track"]), auto_reset=False)
    agent = ActorCritic(env.obs_dim, env.action_dim, state["ppo_config"]["hidden"]).to(env.torch_device)
    agent.load_state_dict(state["agent"])
    agent.eval()
    # Evaluation owns separate RNG/environment state, so training is unaffected.
    times, poses, velocities, controls = [], [], [], []
    lidar_poses, ranges, masks = [], [], []
    total_return, progress = 0.0, 0.0
    reason = 0

    def record(timestamp):
        # Host recordings must wait for the Torch stream used by Warp kernels.
        if env.device.is_cuda:
            torch.cuda.current_stream(env.torch_device).synchronize()
        q, qd = env.sim.snapshot()
        lp, lr, lm = env.sim.lidar.snapshot()
        times.append(timestamp)
        poses.append(q); velocities.append(qd)
        controls.append(env.sim.applied_controls.numpy()[0])
        lidar_poses.append(lp); ranges.append(lr); masks.append(lm)

    record(0.0)
    with torch.no_grad():
        for frame in range(episode.max_steps):
            action = agent.sample(env.observation, deterministic=True)[0]
            env.step(action)
            total_return += env.tensors["reward"][0].item()
            reason = int(env.tensors["reason"][0].item())
            if reason:
                progress = env.tensors["finished_laps"][0].item()
            else:
                progress = env.tensors["progress"][0].item() / (2*np.pi)
            if (frame+1) % 2 == 0 or reason:
                record((frame+1)/60)
            if reason:
                break
    diagnostics = env.sim.lidar.diagnostics()
    if diagnostics["iteration_limit_events"]:
        raise RuntimeError(f"Evaluation exhausted LiDAR rays: {diagnostics}")
    metrics = {"checkpoint": str(args.checkpoint), "training_iteration": state["iteration"],
               "map": env.track.name if isinstance(env.track, MapTrack) else "oval",
               "return": total_return, "lap_progress": progress, "successful_lap": reason == 2,
               "termination_reason": reason, "transitions": frame+1, "simulated_seconds": (frame+1)/60,
               "march_diagnostics": diagnostics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    if poses:
        from .simulation import Trajectory
        from .lidar import LidarRecording
        from .playback import save_html
        scans = LidarRecording(np.array(times), np.array(lidar_poses), np.array(ranges), np.array(masks), env.sim.lidar.rays.directions)
        trajectory = Trajectory(np.array(times), np.array(poses), np.array(velocities), metrics, np.array(controls), scans)
        np.savez_compressed(args.output.with_suffix(".npz"), times=trajectory.times, poses=trajectory.poses,
                            velocities=trajectory.velocities, actions=trajectory.actions,
                            lidar_poses=scans.poses, lidar_ranges=scans.ranges, lidar_valid=scans.valid)
        save_html(trajectory, env.track, env.sim.vehicle, args.output.with_suffix(".html"))
        print(f"Replay: {args.output.with_suffix('.html')}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
