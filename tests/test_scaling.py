"""Cached sampling, identical moving poses, and indexing at PPO-sized batches."""
import json

import numpy as np
import pytest
import warp as wp

from racesense3d.sensors import Rays
from warptracer.benchmark import main, make_runner, validate_pair
from warptracer.distance_field import DistanceField, GridScene
from warptracer.lidar import LidarConfig
from warptracer.terrain import OvalTrack


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cached", [False, True])
def test_scale_grazing_ray_converges_beyond_old_budget(device, cached):
    # Car 1526, beam 64, transition 39 of the 4096-car navigation workload.
    # This shallow ray approaches the rising road slowly; 512 steps truncate
    # a real hit. Preserve conservative jumps and the 2 mm surface tolerance.
    scene = GridScene(DistanceField.oval(OvalTrack()), device)
    direction = np.array([[-.9997666, -.00155008, .02155395]], np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    rays = Rays(direction, np.ones(1), (1,))
    position = [[4.60247, 2.9986153, .5830985]]
    limited = scene.sensor(rays, near=0, far=30, max_steps=512, cache_samples=cached)
    assert not limited.scan(position).numpy()[1].any()
    assert limited.diagnostics()["iteration_limit_events"] == 1
    sensor = scene.sensor(rays, near=0, far=30, cache_samples=cached)
    values, valid = sensor.scan(position).numpy()
    assert valid.all() and values[0, 0] == pytest.approx(5.347, abs=.01)
    diagnostics = sensor.diagnostics()
    assert diagnostics["max_steps"] == 2048
    assert 512 < diagnostics["max_iterations_observed"] < 2048
    assert diagnostics["iteration_limit_events"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_cached_sampling_preserves_tilted_scans_and_budget_counts(device):
    track = OvalTrack()
    scene = GridScene(DistanceField.oval(track), device)
    poses = np.array([track.pose(a, .35) for a in np.linspace(-np.pi, np.pi, 257)])
    rotations = np.array([np.array(wp.quat_to_matrix(wp.quat(*q[3:]))).reshape(3, 3) for q in poses])
    # Pitch half the rays' frames to include ground, roof and sky geometry.
    rotations[::2] = rotations[::2] @ np.array(wp.quat_to_matrix(wp.quat_rpy(0., .2, 0.))).reshape(3, 3)
    sensors = [scene.sensor(LidarConfig().rays(), batch_size=len(poses), cache_samples=cache)
               for cache in (False, True)]
    outputs = [sensor.scan(poses[:, :3], rotations).numpy() for sensor in sensors]
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=2e-6, atol=2e-5)
    diagnostics = [sensor.diagnostics() for sensor in sensors]
    assert all(s["iteration_limit_events"] == 0 for s in diagnostics)
    assert diagnostics[0]["max_iterations_observed"] == diagnostics[1]["max_iterations_observed"]


def test_moving_sensor_poses_are_identical_across_geometry_and_sampling():
    runners = [make_runner("moving_scan", "graph", "cpu", 4, num_envs=3, integrator="fused",
                           track="oval", lidar_backend=backend, grid_sample_cache=cache)
               for backend, cache in (("mesh", False), ("grid", False), ("grid", True))]
    initial = runners[0].sim.snapshot()[0]
    for step in range(8):
        for runner in runners:
            runner.advance()
        poses = [r.sim.lidar.snapshot()[0] for r in runners]
        np.testing.assert_array_equal(poses[0], poses[1])
        np.testing.assert_array_equal(poses[0], poses[2])
        assert all(r.sim.steps == 4 * (step + 1) for r in runners)
    assert not np.allclose(initial, runners[0].sim.snapshot()[0])
    assert not np.allclose(poses[0][0], poses[0][1])
    for runner in runners:
        runner.clock.assign(np.array([2047 * runner.substeps], np.int32))
        runner.sim.steps = 2047 * runner.substeps
        runner.advance()
        np.testing.assert_array_equal(initial, runner.sim.snapshot()[0])
        ptr = runner.path.ptr
        runner.reset()
        np.testing.assert_array_equal(initial, runner.sim.snapshot()[0])
        assert runner.path.ptr == ptr and runner.sim.lidar.timestamp == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("substeps", [4, 6])
def test_moving_scan_capture_and_reset_parity(device, substeps):
    def runner(backend, integrator):
        return make_runner("moving_scan", backend, device, substeps, num_envs=3,
                           integrator=integrator, track="oval", lidar_backend="grid", grid_sample_cache=True)
    candidate = runner("graph", "fused")
    validate_pair(runner("eager", "unfused"), candidate, transitions=70)
    assert candidate.sim.steps == 0
    assert candidate.sim.lidar.diagnostics()["iteration_limit_events"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_4096_moving_scans_match_independent_entries(device):
    options = dict(integrator="fused", track="oval", lidar_backend="grid", grid_sample_cache=True)
    batch = make_runner("moving_scan", "graph", device, 4, num_envs=4096, **options)
    car = make_runner("moving_scan", "graph", device, 4, num_envs=1, **options)
    for _ in range(2):
        batch.advance()
    q, qd = batch.sim.snapshot()
    poses, ranges, valid = batch.sim.lidar.snapshot()
    for index in (0, 31, 256, 1023, 4095):
        car.offsets.assign(np.array([index], np.int32))
        car.sim.initial_pose[:] = batch.sim.initial_pose[index]
        car.reset()
        for _ in range(2):
            car.advance()
        single_q, single_qd = car.sim.snapshot()
        single_pose, single_ranges, single_valid = car.sim.lidar.snapshot()
        np.testing.assert_allclose(q[index], single_q, atol=2e-5)
        np.testing.assert_allclose(qd[index], single_qd, atol=2e-5)
        np.testing.assert_allclose(poses[index], single_pose, atol=2e-5)
        np.testing.assert_allclose(ranges[index], single_ranges, atol=2e-5)
        np.testing.assert_array_equal(valid[index], single_valid)
    assert batch.sim.lidar.diagnostics()["iteration_limit_events"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_4096_navigation_matches_independent_phases(device):
    options = dict(integrator="fused", track="oval", lidar_backend="grid", spread_spawns=True)
    batch = make_runner("navigation", "graph", device, 4, num_envs=4096, **options)
    car = make_runner("navigation", "graph", device, 4, num_envs=1, **options)
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


def test_scale_preset_reports_controlled_scans_and_cache_comparisons(tmp_path):
    output = tmp_path / "scale.json"
    main(["scale", "--device", "cpu", "--envs", "3", "--seconds", ".1", "--trials", "1",
          "--warmup-seconds", ".1", "--warmup-wall-seconds", "0", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["schema_version"] == 5 and report["preset"] == "scale"
    assert report["grid_sampling"] == "both" and len(report["results"]) == 6
    assert report["spread_navigation_spawns"]
    assert len(report["lidar_speedups"]) == 4 and len(report["cache_speedups"]) == 2
    assert all(v["status"] == "passed" for v in report["validation"])
    for result in report["results"]:
        assert result["spawn_distribution"] == "uniform centerline phases"
        diagnostics = result["trials"][0]["march_diagnostics"]
        if diagnostics:
            assert diagnostics["sample_cache"] == result["grid_sample_cache"]
            assert diagnostics["iteration_limit_events"] == 0
        if result["case"] == "moving_scan":
            assert result["controller"] == "manual"
            assert result["physics_substeps_per_transition"] == 0
            assert result["summary"]["median_physics_substeps_per_second"] == 0
            assert result["summary"]["median_simulated_seconds_per_second_per_env"] is None
        else:
            assert result["controller"] == "disparity"
