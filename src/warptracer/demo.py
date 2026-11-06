"""Run from the repository root: uv run warptracer demo."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .simulation import Simulation
from .lidar import LidarConfig
from .disparity import DisparityConfig
from .execution import TransitionRunner
from .terrain import OvalTrack
from .layers import figure_eight_bridge
from .playback import play, save_recording

SCENARIOS = ("oval", "oval-grid", "bridge", "disparity", "accelerate-brake", "circle", "s-turn")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="warptracer demo",
                                     description="30-second LiDAR navigation replay; choose a scenario or override a setting")
    parser.add_argument("scenario", nargs="?", choices=SCENARIOS, default="disparity")
    parser.add_argument("--backend", choices=("eager", "graph"), default="graph", help="Execution backend for disparity navigation")
    parser.add_argument("--max-speed", type=float, default=1.5, help="Disparity target speed limit in m/s")
    parser.add_argument("--device", default=None, help="cpu or cuda:0; default selects available CUDA")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--physics-hz", type=int, default=240)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--lidar-beams", type=int, default=LidarConfig.beams, help="Rays per scan (default: 108)")
    parser.add_argument("--lidar-hz", type=int, default=60, help="Scan rate (default: 60), must divide physics Hz")
    parser.add_argument("--lidar-backend", choices=("mesh", "grid"), default=None, help="oval-grid selects grid; other demos select mesh")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--view", action="store_true", help="Open the recorded run in Warp's native viewer")
    display.add_argument("--video", action="store_true", help="Render an MP4 with Warp (Colab/offscreen)")
    parser.add_argument("--follow", nargs="?", const=20.0, type=float, default=None,
                        help="Top-down camera tracking the car instead of a fixed wide shot; "
                             "optional meters of altitude (default 20); useful on large tracks")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    oval = args.scenario in ("oval", "oval-grid")
    bridge = args.scenario == "bridge"
    navigation = args.scenario == "disparity" or oval or bridge
    lidar_backend = args.lidar_backend or ("grid" if args.scenario == "oval-grid" else "mesh")
    if lidar_backend == "grid" and not oval:
        parser.error("Grid LiDAR currently requires the oval or oval-grid demo")
    sim = Simulation(track=OvalTrack() if oval else figure_eight_bridge() if bridge else None,
                     scenario="drive" if navigation else args.scenario, device=args.device, physics_hz=args.physics_hz,
                     lidar=LidarConfig(beams=args.lidar_beams, frequency=args.lidar_hz, backend=lidar_backend))
    duration = args.seconds if args.seconds is not None else 120.0 if oval or bridge else 30.0 if navigation else 10.0
    runner = None
    if navigation:
        runner = TransitionRunner(sim, controller="disparity", backend=args.backend, integrator="fused",
                                  substeps=sim.lidar_stride, disparity_config=DisparityConfig(max_speed=args.max_speed))
    trajectory = sim.run(duration, args.record_fps, runner=runner)
    if runner is not None:
        trajectory.metadata["disparity"] = asdict(runner.navigator.config)
    if oval:
        a, b = sim.track.center_axes
        phase = np.unwrap(np.arctan2(trajectory.poses[:, 1] / b, trajectory.poses[:, 0] / a))
        progress = max(0.0, float((phase[-1] - phase[0]) / (2 * np.pi)))
    elif bridge:
        progress = max(0.0, sim.track.route_progress(trajectory.poses))
    if oval or bridge:
        trajectory.metadata["lap_progress"] = progress
        trajectory.metadata["completed_laps"] = int(progress)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / args.scenario
    recording = save_recording(trajectory, sim.track, sim.vehicle, stem)
    stem.with_suffix(".json").write_text(json.dumps(trajectory.metadata, indent=2) + "\n")
    print(f"Recording: {recording}; view with: uv run warptracer view {recording}")
    print(f"{trajectory.metadata['simulated_seconds']:.2f} simulated seconds; "
          f"{trajectory.metadata['elapsed_seconds']:.3f} s elapsed; "
          f"{trajectory.metadata['steps_per_second']:.0f} physics steps/s on {sim.device}.")
    speed = np.linalg.norm(trajectory.velocities[:, :2], axis=1)
    print(f"Recorded speed: peak {speed.max():.2f} m/s; final {speed[-1]:.3f} m/s.")
    scans = trajectory.lidar
    print(f"LiDAR: {lidar_backend}; {scans.ranges.shape[1]} beams at {args.lidar_hz} Hz; "
          f"{len(scans.times)} saved scans; {100 * scans.valid.mean():.1f}% valid returns.")
    if oval:
        print(f"Completed laps: {trajectory.metadata['completed_laps']}; elevation range: {np.ptp(trajectory.poses[:, 2]):.3f} m.")
    elif bridge:
        elevated = trajectory.poses[:, 2] > sim.drive.ride_height(sim.vehicle) + .35
        print(f"Completed laps: {trajectory.metadata['completed_laps']}; "
              f"{100 * elevated.mean():.0f}% of samples on the upper half of the {sim.track.route[:, 2].max():.2f} m bridge.")
    if runner is not None:
        print(f"Wall-contact substeps: {trajectory.metadata['wall_contact_substeps']}")
    if args.view or args.video:
        try:
            path = play(trajectory, sim.track, sim.vehicle, video=stem.with_suffix(".mp4") if args.video else None,
                        device=sim.device, follow_height=args.follow)
        except (RuntimeError, OSError) as error:
            parser.error(f"Recording saved to {recording}. {error}")
        if path:
            print(f"Warp video: {path}")
