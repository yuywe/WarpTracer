"""Repeatable single/batched-car throughput measurements; run with warptracer benchmark."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import warp as wp

from .execution import TransitionRunner
from .lidar import LidarConfig
from .simulation import Simulation
from .terrain import OvalTrack


CASES = ("physics", "lidar", "recording", "navigation")

# Related settings live together so normal runs need only a preset name.
PRESETS = {
    "quick": dict(device=None, backend="both", integrator="fused",
                  envs=[1], cases=["physics", "lidar", "recording"], seconds=10),
    "fusion": dict(device="cuda:0", backend="graph", integrator="both",
                   envs=[1], cases=["physics", "lidar"], seconds=1000),
    "batches": dict(device="cuda:0", backend="graph", integrator="fused",
                    envs=[1, 64, 256, 1024], cases=["physics", "lidar"], seconds=100),
    "navigation": dict(device="cuda:0", backend="graph", integrator="fused",
                       envs=[1, 64, 256], cases=["lidar", "navigation"], seconds=100),
    "oval": dict(device="cuda:0", backend="graph", integrator="fused", track="oval",
                 envs=[1, 64, 256], cases=["navigation"], seconds=100,
                 output=Path("outputs/oval-benchmark.json")),
    "grid": dict(device="cuda:0", backend="graph", integrator="fused", track="oval",
                 lidar_backend="both", envs=[1, 64, 256], cases=["navigation"], seconds=100),
}


def make_runner(case, backend, device, substeps, beams=LidarConfig.beams, num_envs=1, integrator="unfused",
                track="room", lidar_backend="mesh", grid_cell_size=.025):
    lidar = None if case == "physics" else LidarConfig(beams=beams, frequency=240 // substeps,
                                                       backend=lidar_backend, grid_cell_size=grid_cell_size)
    sim = Simulation(track=OvalTrack() if track == "oval" else None, scenario="circle", device=device,
                     lidar=lidar, num_envs=num_envs)
    return TransitionRunner(sim, backend=backend, substeps=substeps,
                            controller="disparity" if case == "navigation" else "circle", integrator=integrator)


def validate_pair(eager, graph, transitions=90):
    """Fail before timing if an execution variant changes any car or scan."""
    results = []
    for runner in (eager, graph):
        runner.reset()
        for _ in range(transitions):
            runner.advance()
        wp.synchronize_device(runner.sim.device)
        sim = runner.sim
        q, qd = sim.snapshot()
        result = [q, qd, sim.applied_controls.numpy().copy(), sim.lean_motion.numpy().copy(),
                  sim.collision.numpy().copy(), sim.wall_contact_substeps.numpy().copy()]
        if sim.layered:
            result.append(sim.layers.numpy().copy())
        if runner.navigator is not None:
            result.extend([runner.navigator.filtered.numpy().copy(), runner.navigator.goals.numpy().copy(),
                           runner.navigator.turns.numpy().copy(), sim.device_commands.numpy().copy()])
        if sim.lidar is not None:
            pose, values, valid = sim.lidar.snapshot()
            result.extend([pose, values, valid])
        assert int(runner.clock.numpy()[0]) == transitions * runner.substeps
        results.append(result)
    for a, b in zip(*results):
        if a.dtype == np.bool_:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-4)
    eager.reset()
    graph.reset()


def warm_up(runner, min_transitions, min_wall_seconds, transitions_per_chunk=None):
    """Run actual work until both simulated-step and wall-clock minima are met."""
    runner.reset()
    wp.synchronize_device(runner.sim.device)
    start = time.perf_counter()
    count = 0
    chunk = min_transitions if transitions_per_chunk is None else transitions_per_chunk
    if chunk < 1:
        raise ValueError("Warmup chunk must contain at least one transition")
    while count < min_transitions or time.perf_counter() - start < min_wall_seconds:
        for _ in range(chunk):
            runner.advance()
        count += chunk
        wp.synchronize_device(runner.sim.device)
    elapsed = time.perf_counter() - start
    runner.reset()
    wp.synchronize_device(runner.sim.device)
    return count, elapsed


def trial(runner, case, transitions, warmup, record_hz, warmup_wall_seconds=2.0):
    sim = runner.sim
    transition_hz = sim.physics_hz // runner.substeps
    record_stride = transition_hz // record_hz
    warmup_count, warmup_elapsed = warm_up(runner, warmup, warmup_wall_seconds,
                                          transitions_per_chunk=transitions)
    if sim.lidar is not None:
        # Exclude warmup and the reset's initial scan from convergence counts.
        sim.lidar.reset_statistics()
        wp.synchronize_device(sim.device)
    records = []

    def sample():
        q, qd = sim.snapshot()
        controls = sim.applied_controls.numpy().copy()
        scan = sim.lidar.snapshot() if sim.lidar is not None else None
        records.append((q, qd, controls, scan))

    if case == "recording":
        sample()
    with wp.ScopedDevice(sim.device):
        start = time.perf_counter()
        for i in range(1, transitions + 1):
            runner.advance()
            if case == "recording" and i % record_stride == 0:
                sample()
        wp.synchronize_device(sim.device)
        elapsed = time.perf_counter() - start
    # Inspection and correctness checks are outside the timed region.
    q, qd = sim.snapshot()
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise RuntimeError("Benchmark produced a non-finite vehicle state")
    if sim.steps != transitions * runner.substeps:
        raise RuntimeError("Benchmark step count mismatch")
    if case == "recording" and not all(np.isfinite(item[0]).all() for item in records):
        raise RuntimeError("Recording contains non-finite poses")
    diagnostics = sim.lidar.diagnostics() if sim.lidar is not None else None
    if diagnostics and diagnostics["iteration_limit_events"]:
        raise RuntimeError(f"Distance-field benchmark exhausted the march budget: {diagnostics}")
    return {
        "elapsed_seconds": elapsed,
        "warmup_transitions": warmup_count, "warmup_wall_seconds": warmup_elapsed,
        "physics_substeps_per_second": sim.num_envs * transitions * runner.substeps / elapsed,
        "aggregate_environment_transitions_per_second": sim.num_envs * transitions / elapsed,
        "batch_transitions_per_second": transitions / elapsed,
        "simulated_seconds_per_second": transitions * runner.substeps * sim.dt / elapsed,
        "recorded_frames": len(records),
        "checked_environments": sim.num_envs,
        "march_diagnostics": diagnostics,
        "wall_contact_substeps": sim.wall_contact_substeps.numpy().tolist(),
    }


def summarize(samples, transitions, substeps, num_envs=1):
    times = [sample["elapsed_seconds"] for sample in samples]
    median = statistics.median(times)
    spread = max(times) / min(times)
    return {
        "median_seconds": median, "min_seconds": min(times), "max_seconds": max(times),
        "max_min_ratio": spread,
        "timing_variable": spread > 1.2,
        "median_physics_substeps_per_second": num_envs * transitions * substeps / median,
        "median_aggregate_environment_transitions_per_second": num_envs * transitions / median,
        "median_batch_transitions_per_second": transitions / median,
        "median_simulated_seconds_per_second_per_env": transitions * substeps / 240 / median,
    }


# --- Command-line interface ------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(prog="warptracer benchmark",
                                     description="Choose a preset; add options only to override its settings")
    parser.add_argument("preset", nargs="?", choices=PRESETS, default="quick",
                        help="quick: execution/replay; fusion: integration; batches: scaling; navigation: room control; oval: track control; grid: mesh vs EDT LiDAR")
    parser.add_argument("--track", choices=("room", "oval"), default="room", help="Preset chooses the track")
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--envs", nargs="+", type=int, help="Independent car counts to sweep")
    parser.add_argument("--device", default=None, help="GPU presets require CUDA; quick selects CUDA if available, else CPU")
    parser.add_argument("--backend", choices=("eager", "graph", "both"))
    parser.add_argument("--integrator", choices=("unfused", "fused", "both"))
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per transition")
    parser.add_argument("--lidar-beams", type=int, default=LidarConfig.beams)
    parser.add_argument("--lidar-backend", choices=("mesh", "grid", "both"), default="mesh")
    parser.add_argument("--grid-cell-size", type=float, default=.025, help="Grid resolution in meters; default 2.5 cm")
    parser.add_argument("--seconds", type=float, help="Simulated seconds per car per trial")
    parser.add_argument("--warmup-seconds", type=float, default=2, help="Minimum simulated warmup seconds")
    parser.add_argument("--warmup-wall-seconds", type=float, default=2,
                        help="Minimum real warmup seconds before EACH trial; excluded from timing")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--record-hz", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser


def _parse_args(argv):
    """Parse options, then let the chosen preset supply defaults for anything not explicitly set."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    parser.set_defaults(**{"output": Path("outputs") / f"{args.preset}.json", **PRESETS[args.preset]})
    return parser, parser.parse_args(argv)  # Explicit options override preset defaults.


def _validate(parser, args):
    if args.trials < 1 or not np.isfinite((args.seconds, args.warmup_seconds, args.warmup_wall_seconds)).all():
        parser.error("Require positive trials and finite durations")
    if args.seconds <= 0 or args.warmup_seconds <= 0 or args.warmup_wall_seconds < 0:
        parser.error("Measured/simulated warmup durations must be positive; wall warmup must be nonnegative")
    if args.substeps < 1 or 240 % args.substeps:
        parser.error("substeps must divide 240")
    if args.lidar_beams < 1 or min(args.envs) < 1:
        parser.error("lidar-beams and envs must be positive")
    if args.lidar_backend != "mesh":
        if args.track != "oval":
            parser.error("Grid LiDAR currently requires --track oval (or the grid preset)")
        if not np.isfinite(args.grid_cell_size) or not 0 < args.grid_cell_size <= OvalTrack().barrier_thickness / 4:
            parser.error("Oval grid cell size must be finite, positive and <= barrier_thickness/4 (0.03 m)")
    hz = 240 // args.substeps
    if args.record_hz <= 0 or ("recording" in args.cases and hz % args.record_hz):
        parser.error("record-hz must be positive and divide transition frequency for recording")
    if min(round(args.seconds * hz), round(args.warmup_seconds * hz)) < 1:
        parser.error("Durations must each span at least one transition")


def _device(parser, args):
    wp.init()
    try:
        return wp.get_device(args.device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
    except ValueError:
        parser.error(f"Device {args.device!r} is unavailable. In Colab select Runtime > Change runtime type > GPU, "
                     "or use --device cpu for a CPU check (reduce --envs for large presets).")


def _both(value, options):
    return list(options) if value == "both" else [value]


def _build_runners(args, device, hz):
    """One runner per (cars, case, integrator, backend, LiDAR); validate each against unfused eager."""
    runners, validations = {}, []
    for count in dict.fromkeys(args.envs):
        for case in dict.fromkeys(args.cases):
            for lidar_backend in (_both(args.lidar_backend, ("mesh", "grid")) if case != "physics" else ["mesh"]):
                def build(backend, integrator):
                    return make_runner(case, backend, device, args.substeps, args.lidar_beams, count, integrator,
                                       args.track, lidar_backend, args.grid_cell_size)
                reference = build("eager", "unfused")
                for integrator in _both(args.integrator, ("unfused", "fused")):
                    for backend in _both(args.backend, ("eager", "graph")):
                        key = (count, case, integrator, backend, lidar_backend)
                        is_reference = (integrator, backend) == ("unfused", "eager")
                        runners[key] = reference if is_reference else build(backend, integrator)
                        if not is_reference:
                            validate_pair(reference, runners[key], transitions=max(hz, 2))
                            validations.append({"environments": count, "case": case, "integrator": integrator,
                                                "backend": backend, "lidar_backend": lidar_backend, "status": "passed"})
                            print(f"Unfused/eager parity passed: {key}", flush=True)
    return runners, validations


def _run_trials(runners, args, transitions, warmup):
    """Repeat every runner `--trials` times, rotating the order to spread any drift."""
    samples = {key: [] for key in runners}
    keys = list(runners)
    for repeat in range(args.trials):
        order = keys[repeat % len(keys):] + keys[:repeat % len(keys)]
        for key in order[::-1] if repeat % 2 else order:
            count, case, integrator, backend, lidar_backend = key
            result = trial(runners[key], case, transitions, warmup, args.record_hz, args.warmup_wall_seconds)
            samples[key].append(result)
            print(f"trial {repeat+1}/{args.trials} N={count} {case} {lidar_backend} {integrator}/{backend}: "
                  f"{result['elapsed_seconds']:.3f}s; "
                  f"{result['aggregate_environment_transitions_per_second']:.0f} aggregate transitions/s; "
                  f"{result['batch_transitions_per_second']:.0f} batch transitions/s", flush=True)
    return samples


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser, args = _parse_args(argv)
    _validate(parser, args)
    device = _device(parser, args)
    hz = 240 // args.substeps
    transitions, warmup = round(args.seconds * hz), round(args.warmup_seconds * hz)
    print(f"Preset: {args.preset}; report: {args.output}", flush=True)
    print(f"Device: {device.name}; environments: {args.envs}; {args.substeps} substeps/transition; "
          f"{hz} transitions per simulated second; {args.lidar_beams} LiDAR rays ({args.lidar_backend}).", flush=True)
    if "graph" in _both(args.backend, ("eager", "graph")) and not device.is_cuda:
        print("CPU API graph replay is not a CUDA performance measurement.", flush=True)
    runners, validations = _build_runners(args, device, hz)
    samples = _run_trials(runners, args, transitions, warmup)
    report = {
        "preset": args.preset, "device": str(device), "device_name": device.name,
        "platform": platform.platform(), "python": platform.python_version(),
        "environment_counts": list(dict.fromkeys(args.envs)), "track_kind": args.track,
        "track": asdict(next(iter(runners.values())).sim.track),
        "integrators": _both(args.integrator, ("unfused", "fused")), "physics_hz": 240,
        "substeps_per_transition": args.substeps, "lidar_hz": hz, "lidar_beams": args.lidar_beams,
        "simulated_seconds_per_trial": transitions / hz, "validation": validations,
        "timing_scope": "End-to-end Python submission, GPU waiting, stepping, sensors and optional host recording; excludes setup, warmup, reset, validation, rendering and disk writes",
        "results": [],
    }
    for key, runner in runners.items():
        count, case, integrator, backend, lidar_backend = key
        summary = summarize(samples[key], transitions, args.substeps, count)
        report["results"].append({
            "environments": count, "case": case, "integrator": integrator, "backend": backend,
            "lidar_backend": lidar_backend if case != "physics" else None,
            "lidar_algorithm": runner.sim.lidar.algorithm if runner.sim.lidar else None,
            "controller": runner.controller,
            "controller_config": asdict(runner.navigator.config) if runner.navigator else None,
            "summary": summary, "trials": samples[key]})
        print(f"MEDIAN N={count} {case} {lidar_backend} {integrator}/{backend}: "
              f"{summary['median_aggregate_environment_transitions_per_second']:.0f} aggregate transitions/s; "
              f"{summary['median_batch_transitions_per_second']:.0f} batch transitions/s"
              + (" [variable timing: max/min > 1.2]" if summary["timing_variable"] else ""), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Benchmark report: {args.output}", flush=True)
