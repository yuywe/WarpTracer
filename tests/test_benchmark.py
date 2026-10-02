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
