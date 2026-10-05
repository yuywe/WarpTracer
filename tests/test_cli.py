"""Common commands must select usable workloads and accept focused overrides."""
import json

import numpy as np
import pytest
import warp as wp

from warptracer.cli import main


def short_benchmark(output):
    return ["--device", "cpu", "--seconds", ".1", "--trials", "1",
            "--warmup-wall-seconds", "0", "--warmup-seconds", ".1", "--output", str(output)]


def test_default_benchmark_compares_execution_and_records(tmp_path):
    output = tmp_path / "quick.json"
    main(["benchmark", *short_benchmark(output)])
    report = json.loads(output.read_text())
    assert report["preset"] == "quick"
    assert report["environment_counts"] == [1]
    assert report["integrators"] == ["fused"]
    assert report["lidar_beams"] == 108 and report["lidar_hz"] == 60
    assert len(report["results"]) == 6
    assert {r["backend"] for r in report["results"]} == {"eager", "graph"}
    recorded = [r for r in report["results"] if r["case"] == "recording"]
    assert len(recorded) == 2
    assert all(r["trials"][0]["recorded_frames"] == 4 for r in recorded)


def test_navigation_preset_and_explicit_overrides(tmp_path):
    output = tmp_path / "navigation.json"
    # A preset remains effective even when options precede its name.
    main(["benchmark", "--device", "cpu", "navigation", *short_benchmark(output), "--envs", "2"])
    report = json.loads(output.read_text())
    assert report["preset"] == "navigation"
    assert report["environment_counts"] == [2]
    assert report["simulated_seconds_per_trial"] == .1
    assert len(report["results"]) == 2
    assert {r["case"] for r in report["results"]} == {"lidar", "navigation"}
    assert all(r["backend"] == "graph" and r["integrator"] == "fused" for r in report["results"])
    nav = next(r for r in report["results"] if r["case"] == "navigation")
    assert nav["controller"] == "disparity"
    assert nav["trials"][0]["wall_contact_substeps"] == [0, 0]


@pytest.mark.parametrize("preset,variants", [("fusion", 2), ("batches", 1)])
def test_other_presets_preserve_their_execution_choices(tmp_path, preset, variants):
    output = tmp_path / f"{preset}.json"
    main(["benchmark", preset, *short_benchmark(output), "--envs", "1", "--cases", "physics"])
    report = json.loads(output.read_text())
    assert report["preset"] == preset
    assert len(report["results"]) == variants
    assert all(r["backend"] == "graph" for r in report["results"])
    assert report["integrators"] == (["unfused", "fused"] if preset == "fusion" else ["fused"])


def test_demo_command_exports_navigation_without_extra_flags(tmp_path):
    main(["demo", "--seconds", ".1", "--output", str(tmp_path)])
    metadata = json.loads((tmp_path / "disparity.json").read_text())
    arrays = np.load(tmp_path / "disparity.npz")
    assert (tmp_path / "disparity.html").stat().st_size > 1000
    assert metadata["controller"] == "disparity"
    assert metadata["backend"] == "graph" and metadata["integrator"] == "fused"
    assert metadata["lidar"]["frequency"] == 60
    assert arrays["lidar_ranges"].shape == (7, 108)


@pytest.mark.skipif(wp.is_cuda_available(), reason="Needs an environment without CUDA")
def test_gpu_preset_fails_before_simulation_when_cuda_is_missing(capsys):
    with pytest.raises(SystemExit) as err:
        main(["benchmark", "navigation"])
    assert err.value.code == 2
    assert "Change runtime type" in capsys.readouterr().err
