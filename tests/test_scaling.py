"""Grazing-ray convergence and indexing at PPO-sized batches."""
import numpy as np
import pytest
import warp as wp

from racesense3d.sensors import Rays
from warptracer.benchmark import make_runner
from warptracer.distance_field import DistanceField, GridScene
from warptracer.terrain import OvalTrack


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
def test_grazing_ray_converges_beyond_old_budget(device):
    # Car 1526, beam 64, transition 39 of a 4096-car navigation workload.
    # This shallow ray approaches the rising road slowly; 512 steps truncate
    # a real hit. Preserve conservative jumps and the 2 mm surface tolerance.
    scene = GridScene(DistanceField.oval(OvalTrack()), device)
    direction = np.array([[-.9997666, -.00155008, .02155395]], np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    rays = Rays(direction, np.ones(1), (1,))
    position = [[4.60247, 2.9986153, .5830985]]
    limited = scene.sensor(rays, near=0, far=30, max_steps=512)
    assert not limited.scan(position).numpy()[1].any()
    assert limited.diagnostics()["iteration_limit_events"] == 1
    sensor = scene.sensor(rays, near=0, far=30)
    values, valid = sensor.scan(position).numpy()
    assert valid.all() and values[0, 0] == pytest.approx(5.347, abs=.01)
    diagnostics = sensor.diagnostics()
    assert diagnostics["max_steps"] == 2048
    assert 512 < diagnostics["max_iterations_observed"] < 2048
    assert diagnostics["iteration_limit_events"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_4096_navigation_matches_independent_phases(device):
    options = dict(integrator="fused", track="oval", lidar_backend="grid")
    batch = make_runner("navigation", "graph", device, 4, num_envs=4096, **options)
    car = make_runner("navigation", "graph", device, 4, num_envs=1, **options)
    height = batch.sim.drive.ride_height(batch.sim.vehicle)
    batch.sim.initial_pose[:] = [batch.sim.track.pose(angle, height)
                                 for angle in np.linspace(-np.pi / 2, 3 * np.pi / 2, 4096, endpoint=False)]
    batch.reset()
    for _ in range(2):
        batch.advance()
    q, qd = batch.sim.snapshot()
    _, ranges, valid = batch.sim.lidar.snapshot()
    controls = batch.sim.applied_controls.numpy()
    assert np.isfinite(q).all() and np.isfinite(qd).all()
    assert not batch.sim.wall_contact_substeps.numpy().any()
    for index in (0, 257, 1023, 4095):
        car.sim.initial_pose[:] = batch.sim.initial_pose[index]
        car.reset()
        for _ in range(2):
            car.advance()
        single_q, single_qd = car.sim.snapshot()
        _, single_ranges, single_valid = car.sim.lidar.snapshot()
        np.testing.assert_allclose(q[index], single_q, atol=2e-5)
        np.testing.assert_allclose(qd[index], single_qd, atol=2e-5)
        np.testing.assert_allclose(controls[index], car.sim.applied_controls.numpy()[0], atol=2e-5)
        np.testing.assert_allclose(ranges[index], single_ranges, atol=2e-5)
        np.testing.assert_array_equal(valid[index], single_valid)
