"""Run from the repository root: uv run --extra viz warptracer-demo."""
import argparse
import json
from pathlib import Path

import numpy as np

from .simulation import Simulation
from .lidar import LidarConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flat rectangle and one rigid box in Newton")
    parser.add_argument("--scenario", choices=("accelerate-brake", "circle", "s-turn", "drop", "wall-impact"), default="accelerate-brake")
    parser.add_argument("--device", default=None, help="cpu or cuda:0; default selects available CUDA")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--physics-hz", type=int, default=240)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--no-lidar", action="store_true", help="Run the original vehicle-only scene")
    parser.add_argument("--lidar-hz", type=int, default=30, help="Scan rate, must divide physics Hz")
    parser.add_argument("--headless", action="store_true", help="Skip replay and intermediate CPU copies")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    sim = Simulation(scenario=args.scenario, device=args.device, physics_hz=args.physics_hz,
                     lidar=None if args.no_lidar else LidarConfig(frequency=args.lidar_hz))
    duration = args.seconds if args.seconds is not None else (10.0 if sim.driving else 4.0)
    trajectory = sim.run(duration, args.record_fps, record=not args.headless)
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
        print(f"LiDAR: {scans.ranges.shape[1]} beams at {args.lidar_hz} Hz; "
              f"{len(scans.times)} saved scans; {100 * scans.valid.mean():.1f}% valid returns.")
    print("One box chassis, one floor, four walls.")


if __name__ == "__main__":
    main()
