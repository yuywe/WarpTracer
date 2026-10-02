"""Repeatable single-car throughput measurements; run with python -m warptracer.benchmark."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import platform
import pstats
import statistics
import time
from importlib.metadata import version

import numpy as np
import warp as wp

from .execution import TransitionRunner
from .lidar import LidarConfig
from .simulation import Simulation


CASES = ("physics", "lidar", "recording")


def make_runner(case, backend, device, substeps, engine="lean"):
    lidar = None if case == "physics" else LidarConfig(frequency=240 // substeps)
    sim = Simulation(scenario="circle", device=device, lidar=lidar, engine=engine)
    return TransitionRunner(sim, backend=backend, substeps=substeps, controller="circle")


def validate_pair(eager, graph, transitions=90):
    """Fail before timing if graph replay changes state or sensing."""
    results = []
    for runner in (eager, graph):
        runner.reset()
        for _ in range(transitions):
            runner.advance()
        wp.synchronize_device(runner.sim.model.device)
        sim = runner.sim
        q, qd = sim.snapshot()
        result = [q, qd, sim.applied_controls.numpy().copy()]
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


def trial(runner, case, transitions, warmup, record_hz):
    sim = runner.sim
    transition_hz = sim.physics_hz // runner.substeps
    record_stride = transition_hz // record_hz
    runner.reset()
    for _ in range(warmup):
        runner.advance()
    wp.synchronize_device(sim.model.device)
    runner.reset()
    wp.synchronize_device(sim.model.device)
    records = []

    def sample():
        q, qd = sim.snapshot()
        controls = sim.applied_controls.numpy().copy()
        scan = sim.lidar.snapshot() if sim.lidar is not None else None
        records.append((q, qd, controls, scan))

    if case == "recording":
        sample()
    events = None
    if sim.model.device.is_cuda:
        events = [wp.Event(device=sim.model.device, enable_timing=True) for _ in range(2)]
    with wp.ScopedDevice(sim.model.device):
        cpu_start = time.process_time()
        start = time.perf_counter()
        if events:
            wp.record_event(events[0])
        for i in range(1, transitions + 1):
            runner.advance()
            if case == "recording" and i % record_stride == 0:
                sample()
        if events:
            wp.record_event(events[1])
        wp.synchronize_device(sim.model.device)
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - cpu_start
    # Inspection and correctness checks are outside the timed region.
    q, qd = sim.snapshot()
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise RuntimeError("Benchmark produced a non-finite vehicle state")
    if sim.steps != transitions * runner.substeps:
        raise RuntimeError("Benchmark step count mismatch")
    if case == "recording" and not all(np.isfinite(item[0]).all() for item in records):
        raise RuntimeError("Recording contains non-finite poses")
    return {
        "elapsed_seconds": elapsed,
        "cpu_process_seconds": cpu_elapsed,
        # Event intervals include stream idle gaps; these are not summed kernel durations.
        "cuda_stream_seconds": wp.get_event_elapsed_time(*events, synchronize=False) / 1000 if events else None,
        "physics_substeps_per_second": transitions * runner.substeps / elapsed,
        "environment_transitions_per_second": transitions / elapsed,
        "simulated_seconds_per_second": transitions * runner.substeps * sim.dt / elapsed,
        "recorded_frames": len(records),
        "final_pose": q.tolist(), "final_velocity": qd.tolist(),
    }


def summarize(samples, transitions, substeps):
    times = [sample["elapsed_seconds"] for sample in samples]
    median = statistics.median(times)
    return {
        "median_seconds": median, "min_seconds": min(times), "max_seconds": max(times),
        "median_physics_substeps_per_second": transitions * substeps / median,
        "median_environment_transitions_per_second": transitions / median,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-car eager/graph benchmark, with paired validation")
    parser.add_argument("--device", default=None, help="Default: CUDA if available, else CPU")
    parser.add_argument("--physics", choices=("lean", "newton"), default="lean")
    parser.add_argument("--backend", choices=("auto", "eager", "graph", "both"), default="auto")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--substeps", type=int, default=4, help="Even physics substeps per transition")
    parser.add_argument("--record-hz", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    parser.add_argument("--profile", action="store_true", help="Save a separate eager CPU-call profile")
    args = parser.parse_args(argv)
    if args.trials < 1 or not np.isfinite((args.seconds, args.warmup_seconds)).all():
        parser.error("Require positive trials and finite durations")
    if args.seconds <= 0 or args.warmup_seconds <= 0:
        parser.error("Measured duration and warmup must be positive")
    if args.substeps < 2 or args.substeps % 2 or 240 % args.substeps:
        parser.error("substeps must be an even divisor of 240")
    hz = 240 // args.substeps
    if args.record_hz <= 0 or hz % args.record_hz:
        parser.error("record-hz must be a positive divisor of transition frequency")
    transitions = round(args.seconds * hz)
    warmup = round(args.warmup_seconds * hz)
    if min(transitions, warmup) < 1:
        parser.error("Durations must each span at least one transition")
    wp.init()
    device = wp.get_device(args.device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
    backends = (["eager", "graph"] if device.is_cuda else ["eager"]) if args.backend == "auto" else (
        ["eager", "graph"] if args.backend == "both" else [args.backend])
    cases = list(dict.fromkeys(args.cases))
    print(f"Device: {device.name}; physics: {args.physics}; 1 car; {args.substeps} physics substeps/transition; "
          f"{hz} transitions per simulated second.", flush=True)
    print(f"LiDAR cases: 1080 rays once per transition ({hz} Hz). "
          f"{args.trials} trials, each after {warmup / hz:g} simulated seconds of warmup.", flush=True)
    if "graph" in backends and not device.is_cuda:
        print("CPU graph replay validates capture behavior; it is not a CUDA performance measurement.", flush=True)
    runners = {}
    validations = {}
    for case in cases:
        for backend in backends:
            runners[(case, backend)] = make_runner(case, backend, device, args.substeps, args.physics)
        if "graph" in backends:
            eager = runners.get((case, "eager")) or make_runner(case, "eager", device, args.substeps, args.physics)
            validate_pair(eager, runners[(case, "graph")], transitions=max(hz, 2))
            validations[case] = "passed"
            print(f"Eager/graph parity passed: {case}", flush=True)
    samples = {key: [] for key in runners}
    keys = list(runners)
    # Rotate and reverse case order across trials to reduce a fixed ordering bias.
    for repeat in range(args.trials):
        order = keys[repeat % len(keys):] + keys[:repeat % len(keys)]
        if repeat % 2:
            order = order[::-1]
        for case, backend in order:
            result = trial(runners[(case, backend)], case, transitions, warmup, args.record_hz)
            samples[(case, backend)].append(result)
            print(f"trial {repeat+1}/{args.trials} {case:9s} {backend:5s}: "
                  f"{result['elapsed_seconds']:.3f}s; "
                  f"{result['physics_substeps_per_second']:.0f} substeps/s; "
                  f"{result['environment_transitions_per_second']:.0f} transitions/s", flush=True)
    report = {
        "device": str(device), "device_name": device.name, "platform": platform.platform(),
        "python": platform.python_version(), "versions": {p: version(p) for p in ("warp-lang", "newton", "numpy")},
        "physics_engine": args.physics, "environments": 1, "physics_hz": 240, "substeps_per_transition": args.substeps,
        "transition_hz": hz, "lidar_hz": hz, "lidar_beams": 1080,
        "record_hz": args.record_hz, "transitions_per_trial": transitions,
        "simulated_seconds_per_trial": transitions / hz, "warmup_seconds": warmup / hz,
        "graph_validation": validations,
        "timing_scope": "Stepping, sensors, and optional host recording; excludes setup, warmup, reset, validation, HTML and disk writes",
        "results": [],
    }
    for case, backend in keys:
        summary = summarize(samples[(case, backend)], transitions, args.substeps)
        report["results"].append({"case": case, "backend": backend,
                                  "graph_kind": runners[(case, backend)].graph_kind,
                                  "summary": summary, "trials": samples[(case, backend)]})
        print(f"MEDIAN {case:9s} {backend:5s}: {summary['median_physics_substeps_per_second']:.0f} "
              f"substeps/s; {summary['median_environment_transitions_per_second']:.0f} transitions/s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.profile:
        runner = runners.get(("lidar", "eager")) or make_runner("lidar", "eager", device, args.substeps, args.physics)
        profile = cProfile.Profile()
        profile.runcall(trial, runner, "lidar", min(transitions, hz), warmup, args.record_hz)
        profile.dump_stats(str(args.output.with_suffix(".prof")))
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(30)
        args.output.with_suffix(".profile.txt").write_text(stream.getvalue())
    print(f"Benchmark report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
