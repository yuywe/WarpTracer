"""Independent cars and fused substeps must agree with separate single-car runs."""
import numpy as np
import pytest
import warp as wp

from warptracer.benchmark import main, warm_up
from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.scene import Track
from warptracer.simulation import Simulation

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))]


def runner(n, integrator, backend, substeps, device):
    return TransitionRunner(Simulation(scenario="drive", engine="lean", device=device,
        num_envs=n, track=Track(length=4, width=4),
        lidar=LidarConfig(beams=27, frequency=240 // substeps)),
        integrator=integrator, backend=backend, substeps=substeps)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("substeps", [4, 6])
@pytest.mark.parametrize("integrator,backend", [
    ("unfused", "eager"), ("unfused", "graph"), ("fused", "eager"), ("fused", "graph")])
def test_batch_matches_independent_cars_with_controls_collision_and_reset(device, substeps, integrator, backend):
    batch = runner(3, integrator, backend, substeps, device)
    singles = [runner(1, "unfused", "eager", substeps, device) for _ in range(3)]
    phases = [
        np.array([[1, 0, 0, -1], [0, 0, .25, 1], [0, 0, -.25, .7]], np.float32),
        np.array([[0, 1, 0, -1], [0, 0, -.25, .5], [0, 0, .25, 0]], np.float32),
    ]
    for _ in range(2):
        batch.reset()
        for single in singles:
            single.reset()
        for commands in phases:
            batch.sim.set_commands(commands)
            for i, single in enumerate(singles):
                single.sim.set_commands(commands[i:i+1])
            for _ in range(120):
                batch.advance()
                for single in singles:
                    single.advance()
            q, v = batch.sim.snapshot()
            pose, ranges, valid = batch.sim.lidar.snapshot()
            assert q.shape == (3, 7) and ranges.shape == (3, 27)
            for i, single in enumerate(singles):
                sq, sv = single.sim.snapshot()
                sp, sr, sm = single.sim.lidar.snapshot()
                np.testing.assert_allclose(q[i], sq, atol=2e-4, rtol=2e-4)
                np.testing.assert_allclose(v[i], sv, atol=2e-4, rtol=2e-4)
                np.testing.assert_allclose(pose[i], sp, atol=2e-4, rtol=2e-4)
                np.testing.assert_allclose(ranges[i], sr, atol=2e-4, rtol=2e-4)
                np.testing.assert_array_equal(valid[i], sm)
                np.testing.assert_allclose(batch.sim.applied_controls.numpy()[i],
                                           single.sim.applied_controls.numpy()[0], atol=2e-4)
                assert batch.sim.collision.numpy()[i] == single.sim.collision.numpy()[0]
        assert batch.sim.steps == 240 * substeps
        assert batch.sim.lidar.timestamp == pytest.approx(substeps)
        assert not np.allclose(q[0], q[1])
        # First car reached the +X wall, then stayed stopped under braking.
        assert q[0, 0] == pytest.approx(1.74, abs=.001)
        assert np.linalg.norm(v[0]) < .01
        batch.sim.set_action(brake=1)  # scalar inputs broadcast after distinct commands
        batch.advance()
        np.testing.assert_allclose(batch.sim.applied_controls.numpy()[:, :2], [[0, 1]] * 3)


def test_wall_clock_warmup_resets_clock_state_and_scans():
    r = runner(3, "fused", "graph", 4, "cpu")
    before = r.sim.snapshot()[0]
    count, elapsed = warm_up(r, 10, .05)
    assert elapsed >= .05 and count >= 10
    assert r.sim.steps == 0 and r.clock.numpy()[0] == 0
    assert r.sim.lidar.timestamp == 0
    np.testing.assert_array_equal(r.sim.snapshot()[0], before)


def test_sweep_units_and_40hz_headless(tmp_path):
    import json
    output = tmp_path / "sweep.json"
    main(["--device", "cpu", "--backend", "graph", "--integrator", "both",
          "--envs", "1", "3", "--cases", "lidar", "--substeps", "6",
          "--seconds", ".1", "--warmup-seconds", ".1", "--warmup-wall-seconds", "0",
          "--trials", "1", "--output", str(output)])
    data = json.loads(output.read_text())
    assert data["lidar_hz"] == 40
    assert len(data["results"]) == 4 and len(data["validation"]) == 4
    for result in data["results"]:
        summary, n = result["summary"], result["environments"]
        assert summary["median_environment_transitions_per_second"] == pytest.approx(
            n * summary["median_batch_transitions_per_second"])
        assert summary["median_physics_substeps_per_second"] == pytest.approx(
            6 * summary["median_environment_transitions_per_second"])
        assert result["trials"][0]["checked_environments"] == n


def test_invalid_batch_and_newton_fusion():
    with pytest.raises(ValueError, match="Batched"):
        Simulation(scenario="drive", device="cpu", num_envs=2)
    with pytest.raises(ValueError, match="positive"):
        Simulation(scenario="drive", device="cpu", engine="lean", num_envs=0)
    r = runner(3, "fused", "eager", 4, "cpu")
    for commands in ([[0, 0, 0, 1]], [[0, 0, 0, -2]] * 3, [[0, 0, 0, np.nan]] * 3):
        with pytest.raises(ValueError):
            r.sim.set_commands(commands)
    with pytest.raises(ValueError, match="one car"):
        r.sim.run(.1)
    with pytest.raises(ValueError, match="lean"):
        TransitionRunner(Simulation(scenario="drive", device="cpu"), integrator="fused")


@pytest.mark.parametrize("device", DEVICES)
def test_batch_crosses_block_boundary(device):
    n = 257
    batch = runner(n, "fused", "graph", 4, device)
    commands = np.zeros((n, 4), np.float32)
    commands[:, 2] = np.linspace(-.25, .25, n)
    commands[:, 3] = np.linspace(.2, 1, n)
    batch.sim.set_commands(commands)
    for _ in range(60):
        batch.advance()
    q, v = batch.sim.snapshot()
    pose, ranges, valid = batch.sim.lidar.snapshot()
    assert np.isfinite(q).all() and np.isfinite(v).all()
    assert ranges.shape == (n, 27)
    for i in (0, 128, 256):
        single = runner(1, "unfused", "eager", 4, device)
        single.sim.set_commands(commands[i:i+1])
        for _ in range(60):
            single.advance()
        sq, sv = single.sim.snapshot()
        sp, sr, sm = single.sim.lidar.snapshot()
        np.testing.assert_allclose(q[i], sq, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(v[i], sv, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(pose[i], sp, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(ranges[i], sr, atol=2e-4, rtol=2e-4)
        np.testing.assert_array_equal(valid[i], sm)
