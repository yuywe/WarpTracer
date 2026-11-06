"""Captured execution must honor changing inputs, reset, and mounted sensing."""
import numpy as np
import pytest
import warp as wp

from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))])
def test_graph_matches_eager_with_live_commands_and_reset(device):
    runners = [TransitionRunner(Simulation(scenario="drive", device=device,
                   lidar=LidarConfig(frequency=60)), backend=b) for b in ("eager", "graph")]
    for repeat in range(2):
        for r in runners:
            r.reset()
        for speed, steering in [(1, .25), (.7, -.25), (0, 0)]:
            for r in runners:
                r.sim.set_target_speed(speed, steering)
                for _ in range(30):
                    r.advance()
            for a, b in zip(runners[0].sim.snapshot(), runners[1].sim.snapshot()):
                np.testing.assert_allclose(a, b, atol=2e-4, rtol=2e-4)
            for a, b in zip(runners[0].sim.lidar.snapshot(), runners[1].sim.lidar.snapshot()):
                np.testing.assert_allclose(a, b, atol=2e-4, rtol=2e-4)
        assert all(r.sim.steps == 360 for r in runners)
        assert all(r.sim.lidar.timestamp == 1.5 for r in runners)


def test_runner_rejects_bad_substeps_and_sensor_schedule():
    sim = Simulation(scenario="drive", device="cpu", lidar=LidarConfig())
    with pytest.raises(ValueError, match="positive"):
        TransitionRunner(sim, substeps=0)
    with pytest.raises(ValueError, match="LiDAR"):
        TransitionRunner(sim, substeps=4)


@pytest.mark.parametrize("backend", ["eager", "graph"])
def test_odd_substeps_match_single_steps(backend):
    reference = Simulation(scenario="drive", device="cpu")
    candidate = Simulation(scenario="drive", device="cpu")
    for sim in (reference, candidate):
        sim.set_action(throttle=.5, steering=.1)
    runner = TransitionRunner(candidate, backend=backend, substeps=3, integrator="fused")
    candidate.set_action(throttle=.5, steering=.1)
    for _ in range(30):
        runner.advance()
    for _ in range(90):
        reference.step()
    for a, b in zip(reference.snapshot(), candidate.snapshot()):
        np.testing.assert_allclose(a, b, atol=1e-5)
