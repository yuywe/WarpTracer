"""Repeatable single/batched-car throughput measurements; run with python -m warptracer.benchmark."""
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


def make_runner(case, backend, device, substeps, engine="lean", beams=LidarConfig.beams, num_envs=1, integrator="unfused"):
    lidar = None if case == "physics" else LidarConfig(beams=beams, frequency=240 // substeps)
    sim = Simulation(scenario="circle", device=device, lidar=lidar, engine=engine, num_envs=num_envs)
    return TransitionRunner(sim, backend=backend, substeps=substeps, controller="circle", integrator=integrator)


def validate_pair(eager, graph, transitions=90):
    """Fail before timing if an execution variant changes any car or scan."""
    results = []
    for runner in (eager, graph):
        runner.reset()
        for _ in range(transitions):
            runner.advance()
        wp.synchronize_device(runner.sim.model.device)
        sim = runner.sim
        q, qd = sim.snapshot()
        result = [q, qd, sim.applied_controls.numpy().copy()]
        if sim.engine == "lean":
            result.extend([sim.lean_motion.numpy().copy(), sim.collision.numpy().copy()])
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


def warm_up(runner, min_transitions, min_wall_seconds):
    """Run actual work until both simulated-step and wall-clock minima are met."""
    runner.reset()
    wp.synchronize_device(runner.sim.model.device)
    start = time.perf_counter()
    count = 0
    while count < min_transitions or time.perf_counter() - start < min_wall_seconds:
        for _ in range(32):
            runner.advance()
        count += 32
        wp.synchronize_device(runner.sim.model.device)
    elapsed = time.perf_counter() - start
    runner.reset()
    wp.synchronize_device(runner.sim.model.device)
    return count, elapsed


def trial(runner, case, transitions, warmup, record_hz, warmup_wall_seconds=2.0):
    sim = runner.sim
    transition_hz = sim.physics_hz // runner.substeps
    record_stride = transition_hz // record_hz
    warmup_count, warmup_elapsed = warm_up(runner, warmup, warmup_wall_seconds)
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
        "warmup_transitions": warmup_count, "warmup_wall_seconds": warmup_elapsed,
        "physics_substeps_per_second": sim.num_envs * transitions * runner.substeps / elapsed,
        "aggregate_environment_transitions_per_second": sim.num_envs * transitions / elapsed,
        "batch_transitions_per_second": transitions / elapsed,
        "environment_transitions_per_second": sim.num_envs * transitions / elapsed,
        "simulated_seconds_per_second": transitions * runner.substeps * sim.dt / elapsed,
        "recorded_frames": len(records),
        "checked_environments": sim.num_envs,
        "final_pose": (q if sim.num_envs == 1 else q[0]).tolist(),
        "final_velocity": (qd if sim.num_envs == 1 else qd[0]).tolist(),
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
        "median_environment_transitions_per_second": num_envs * transitions / median,
        "median_aggregate_environment_transitions_per_second": num_envs * transitions / median,
        "median_batch_transitions_per_second": transitions / median,
        "median_simulated_seconds_per_second_per_env": transitions * substeps / 240 / median,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vehicle batch, fusion, and graph benchmark with parity validation")
    parser.add_argument("--device", default=None, help="Default: CUDA if available, else CPU")
    parser.add_argument("--physics", choices=("lean", "newton"), default="lean")
    parser.add_argument("--backend", choices=("auto", "eager", "graph", "both"), default="auto")
    parser.add_argument("--integrator", choices=("auto", "unfused", "fused", "both"), default="auto",
                        help="Default: fused for lean, unfused for Newton")
    parser.add_argument("--envs", nargs="+", type=int, default=[1], help="Independent car counts to sweep")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seconds", type=float, default=10, help="Simulated seconds per car per trial")
    parser.add_argument("--warmup-seconds", type=float, default=2, help="Minimum simulated warmup seconds")
    parser.add_argument("--warmup-wall-seconds", type=float, default=2,
                        help="Minimum real warmup seconds before EACH trial; excluded from timing")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--substeps", type=int, default=4, help="Even physics substeps per transition")
    parser.add_argument("--lidar-beams", type=int, default=LidarConfig.beams)
    parser.add_argument("--record-hz", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    parser.add_argument("--profile", action="store_true", help="Separate eager Python profile for the smallest batch")
    args = parser.parse_args(argv)
    if args.trials < 1 or not np.isfinite((args.seconds, args.warmup_seconds, args.warmup_wall_seconds)).all():
        parser.error("Require positive trials and finite durations")
    if args.seconds <= 0 or args.warmup_seconds <= 0 or args.warmup_wall_seconds < 0:
        parser.error("Measured/simulated warmup durations must be positive; wall warmup must be nonnegative")
    if args.substeps < 2 or args.substeps % 2 or 240 % args.substeps:
        parser.error("substeps must be an even divisor of 240")
    if args.lidar_beams < 1 or min(args.envs) < 1:
        parser.error("lidar-beams and envs must be positive")
    integrators = (["fused"] if args.physics == "lean" else ["unfused"]) if args.integrator == "auto" else (
        ["unfused", "fused"] if args.integrator == "both" else [args.integrator])
    env_counts = list(dict.fromkeys(args.envs))
    if args.physics == "newton" and (env_counts != [1] or integrators != ["unfused"]):
        parser.error("Newton supports one environment and unfused integration only")
    hz = 240 // args.substeps
    cases = list(dict.fromkeys(args.cases))
    if args.record_hz <= 0 or ("recording" in cases and hz % args.record_hz):
        parser.error("record-hz must be positive and divide transition frequency for recording")
    transitions = round(args.seconds * hz)
    warmup = round(args.warmup_seconds * hz)
    if min(transitions, warmup) < 1:
        parser.error("Durations must each span at least one transition")
    wp.init()
    device = wp.get_device(args.device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
    backends = (["eager", "graph"] if device.is_cuda else ["eager"]) if args.backend == "auto" else (
        ["eager", "graph"] if args.backend == "both" else [args.backend])
    print(f"Device: {device.name}; physics: {args.physics}; environments: {env_counts}; "
          f"{args.substeps} substeps/transition; {hz} transitions per simulated second.", flush=True)
    print(f"LiDAR: {args.lidar_beams} rays at {hz} Hz. Warmup per trial: at least "
          f"{args.warmup_wall_seconds:g} real seconds AND {warmup / hz:g} simulated seconds.", flush=True)
    if "graph" in backends and not device.is_cuda:
        print("CPU API graph replay is not a CUDA performance measurement.", flush=True)
    runners, validations = {}, []
    for count in env_counts:
        for case in cases:
            reference = make_runner(case, "eager", device, args.substeps, args.physics,
                                    args.lidar_beams, count, "unfused")
            for integrator in integrators:
                for backend in backends:
                    key = (count, case, integrator, backend)
                    runner = (reference if (integrator, backend) == ("unfused", "eager") else
                              make_runner(case, backend, device, args.substeps, args.physics,
                                          args.lidar_beams, count, integrator))
                    runners[key] = runner
                    if runner is not reference:
                        validate_pair(reference, runner, transitions=max(hz, 2))
                        validations.append({"environments": count, "case": case,
                                            "integrator": integrator, "backend": backend,
                                            "reference": "unfused/eager", "status": "passed"})
                        print(f"Unfused/eager parity passed: {key}", flush=True)
    samples = {key: [] for key in runners}
    keys = list(runners)
    for repeat in range(args.trials):
        order = keys[repeat % len(keys):] + keys[:repeat % len(keys)]
        if repeat % 2:
            order = order[::-1]
        for key in order:
            count, case, integrator, backend = key
            result = trial(runners[key], case, transitions, warmup, args.record_hz, args.warmup_wall_seconds)
            samples[key].append(result)
            print(f"trial {repeat+1}/{args.trials} N={count} {case} {integrator}/{backend}: "
                  f"{result['elapsed_seconds']:.3f}s; "
                  f"{result['aggregate_environment_transitions_per_second']:.0f} aggregate transitions/s; "
                  f"{result['batch_transitions_per_second']:.0f} batch transitions/s", flush=True)
    report = {
        "schema_version": 2,
        "device": str(device), "device_name": device.name, "platform": platform.platform(),
        "python": platform.python_version(), "versions": {p: version(p) for p in ("warp-lang", "newton", "numpy")},
        "physics_engine": args.physics, "environment_counts": env_counts,
        "environments": env_counts[0] if len(env_counts) == 1 else None,
        "integrators": integrators, "physics_hz": 240, "substeps_per_transition": args.substeps,
        "transition_hz": hz, "lidar_hz": hz, "lidar_beams": args.lidar_beams,
        "record_hz": args.record_hz, "transitions_per_trial": transitions,
        "simulated_seconds_per_trial": transitions / hz, "warmup_seconds": warmup / hz,
        "minimum_warmup_wall_seconds": args.warmup_wall_seconds,
        "validation": validations,
        "timing_scope": "Stepping, sensors, and optional host recording; excludes setup, warmup, reset, validation, HTML and disk writes",
        "throughput_units": "environment transitions/s and physics substeps/s aggregate all cars; batch transitions/s counts runner advances",
        "final_state_scope": "All cars checked finite; final_pose/final_velocity contain car zero only",
        "results": [],
    }
    for key in keys:
        count, case, integrator, backend = key
        summary = summarize(samples[key], transitions, args.substeps, count)
        report["results"].append({"environments": count, "case": case, "integrator": integrator,
                                  "backend": backend, "graph_kind": runners[key].graph_kind,
                                  "summary": summary, "trials": samples[key]})
        print(f"MEDIAN N={count} {case} {integrator}/{backend}: "
              f"{summary['median_aggregate_environment_transitions_per_second']:.0f} aggregate transitions/s; "
              f"{summary['median_batch_transitions_per_second']:.0f} batch transitions/s"
              + (" [variable timing: max/min > 1.2]" if summary["timing_variable"] else ""), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.profile:
        runner = make_runner("lidar", "eager", device, args.substeps, args.physics,
                             args.lidar_beams, min(env_counts), integrators[0])
        profile = cProfile.Profile()
        profile.runcall(trial, runner, "lidar", min(transitions, hz), warmup, args.record_hz,
                        args.warmup_wall_seconds)
        profile.dump_stats(str(args.output.with_suffix(".prof")))
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(30)
        args.output.with_suffix(".profile.txt").write_text(stream.getvalue())
    print(f"Benchmark report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
