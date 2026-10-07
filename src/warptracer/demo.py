"""Run from the repository root: uv run warptracer demo."""
import argparse
import json
from pathlib import Path

import numpy as np

from .simulation import Simulation
from .lidar import LidarConfig
from .disparity import DisparityConfig
from .execution import TransitionRunner
from .terrain import OvalTrack

SCENARIOS = ("oval", "oval-grid", "disparity", "accelerate-brake", "circle", "s-turn", "drop", "wall-impact")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="warptracer demo",
                                     description="30-second LiDAR navigation replay; choose a scenario or override a setting")
    parser.add_argument("scenario_name", nargs="?", choices=SCENARIOS, help="Default: disparity")
    parser.add_argument("--scenario", choices=SCENARIOS, help="Alias for the positional scenario")
    parser.add_argument("--backend", choices=("eager", "graph"), default="graph", help="Execution backend for disparity navigation")
    parser.add_argument("--max-speed", type=float, default=1.5, help="Disparity target speed limit in m/s")
    parser.add_argument("--physics", choices=("lean", "newton"), default="lean")
    parser.add_argument("--device", default=None, help="cpu or cuda:0; default selects available CUDA")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--physics-hz", type=int, default=240)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--no-lidar", action="store_true", help="Run the original vehicle-only scene")
    parser.add_argument("--lidar-beams", type=int, default=LidarConfig.beams, help="Rays per scan (default: 108)")
    parser.add_argument("--lidar-hz", type=int, default=60, help="Scan rate (default: 60), must divide physics Hz")
    parser.add_argument("--lidar-backend", choices=("mesh", "grid"), default=None, help="oval-grid selects grid; other demos select mesh")
    parser.add_argument("--headless", action="store_true", help="Skip replay and intermediate CPU copies")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if args.scenario_name is not None and args.scenario is not None:
        parser.error("Choose a positional scenario or --scenario, not both")
    args.scenario = args.scenario or args.scenario_name or "disparity"
    oval = args.scenario in ("oval", "oval-grid")
    navigation = args.scenario == "disparity" or oval
    if navigation and args.no_lidar:
        parser.error("Disparity navigation requires LiDAR")
    if oval and args.physics != "lean":
        parser.error("The oval demo currently uses lean road-following physics")
    lidar_backend = args.lidar_backend or ("grid" if args.scenario == "oval-grid" else "mesh")
    if lidar_backend == "grid" and not oval:
        parser.error("Grid LiDAR currently requires the oval or oval-grid demo")
    physics = "newton" if args.scenario in ("drop", "wall-impact") else args.physics
    sim = Simulation(track=OvalTrack() if oval else None,
                     engine=physics, scenario="drive" if navigation else args.scenario, device=args.device, physics_hz=args.physics_hz,
                     lidar=None if args.no_lidar else LidarConfig(beams=args.lidar_beams, frequency=args.lidar_hz, backend=lidar_backend))
    duration = args.seconds if args.seconds is not None else (
        120.0 if oval else 30.0 if navigation else 10.0 if sim.driving else 4.0)
    runner = None
    if navigation:
        runner = TransitionRunner(sim, controller="disparity", backend=args.backend,
                                  integrator="fused" if physics == "lean" else "unfused",
                                  substeps=sim.lidar_stride,
                                  disparity_config=DisparityConfig(max_speed=args.max_speed))
    trajectory = sim.run(duration, args.record_fps, record=not args.headless, runner=runner)
    if runner is not None:
        from dataclasses import asdict
        trajectory.metadata["disparity"] = asdict(runner.navigator.config)
    if sim.oval:
        trajectory.metadata["completed_laps"] = None
        if not args.headless:
            a, b = sim.track.center_axes
            phase = np.unwrap(np.arctan2(trajectory.poses[:, 1] / b, trajectory.poses[:, 0] / a))
            progress = max(0.0, float((phase[-1] - phase[0]) / (2 * np.pi)))
            trajectory.metadata["lap_progress"] = progress
            trajectory.metadata["completed_laps"] = int(progress)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / (args.scenario + ("_headless" if args.headless else ""))
    arrays = dict(times=trajectory.times, poses=trajectory.poses,
                  velocities=trajectory.velocities, actions=trajectory.actions)
    if trajectory.lidar is not None:
        scans = trajectory.lidar
        arrays.update(lidar_times=scans.times, lidar_poses=scans.poses, lidar_ranges=scans.ranges,
                      lidar_valid=scans.valid, lidar_directions=scans.directions)
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text(json.dumps(trajectory.metadata, indent=2) + "\n")
    if not args.headless:
        from .playback import save_html
        path = save_html(trajectory, sim.track, sim.vehicle, stem.with_suffix(".html"))
        print(f"Interactive replay: {path}")
    print(f"{trajectory.metadata['simulated_seconds']:.2f} simulated seconds; "
          f"{trajectory.metadata['elapsed_seconds']:.3f} s elapsed; "
          f"{trajectory.metadata['steps_per_second']:.0f} physics steps/s on {sim.model.device}.")
    speed = np.linalg.norm(trajectory.velocities[:, :2], axis=1)
    if sim.driving and not args.headless:
        print(f"Recorded speed: peak {speed.max():.2f} m/s; final {speed[-1]:.3f} m/s.")
    if trajectory.lidar is not None:
        scans = trajectory.lidar
        print(f"LiDAR: {lidar_backend}; {scans.ranges.shape[1]} beams at {args.lidar_hz} Hz; "
              f"{len(scans.times)} saved scans; {100 * scans.valid.mean():.1f}% valid returns.")
    if sim.oval:
        print(f"Physics: {sim.engine}; one box chassis, one oval road, two continuous barriers.")
        if not args.headless:
            print(f"Completed laps: {trajectory.metadata['completed_laps']}; elevation range: {np.ptp(trajectory.poses[:, 2]):.3f} m.")
    else:
        print(f"Physics: {sim.engine}; one box chassis, one floor, four walls.")
    if runner is not None and physics == "lean":
        print(f"Wall-contact substeps: {trajectory.metadata['wall_contact_substeps']}")


if __name__ == "__main__":
    main()
