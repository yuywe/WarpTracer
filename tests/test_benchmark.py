"""The chosen scan resolution must reach execution and its saved report."""
import json
import pytest
from warptracer.benchmark import main, make_runner


@pytest.mark.parametrize("beams", [108, 1080])
def test_scan_resolution_and_report(tmp_path, beams):
    runner = make_runner("lidar", "graph", "cpu", 4, beams=beams)
    runner.advance()
    assert runner.sim.lidar.snapshot()[1].size == beams
    output = tmp_path / "benchmark.json"
    main(["--device", "cpu", "--backend", "both", "--cases", "recording",
          "--lidar-beams", str(beams), "--seconds", ".1",
          "--warmup-wall-seconds", "0", "--warmup-seconds", ".1", "--trials", "1", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["lidar_beams"] == beams
    assert report["validation"]
    assert all(v["status"] == "passed" for v in report["validation"])
    assert all(r["trials"][0]["recorded_frames"] == 4 for r in report["results"])


def test_navigation_benchmark_reports_controller_and_contacts(tmp_path):
    output = tmp_path / "navigation.json"
    main(["--device", "cpu", "--backend", "both", "--integrator", "both", "--envs", "3",
          "--cases", "navigation", "--seconds", ".1", "--warmup-seconds", ".1",
          "--warmup-wall-seconds", "0", "--trials", "1", "--output", str(output)])
    report = json.loads(output.read_text())
    assert len(report["results"]) == 4
    assert all(v["status"] == "passed" for v in report["validation"])
    for result in report["results"]:
        assert result["controller"] == "disparity"
        assert result["trials"][0]["wall_contact_substeps"] == [0, 0, 0]
