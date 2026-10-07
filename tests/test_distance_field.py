"""Independent surface intersections, EDT bounds, convergence and capture."""
import json

import numpy as np
import pytest
import warp as wp

from racesense3d.sensors import Rays
from warptracer.benchmark import validate_pair
from warptracer.execution import TransitionRunner
from warptracer.distance_field import GridScene, DistanceField, _signed_distance, _gradient_bound
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation
from warptracer.terrain import OvalTrack
from warptracer.lidar_comparison import oval_scan_comparison


def rays(directions):
    d = np.atleast_2d(np.asarray(directions, np.float32))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return Rays(d, np.ones(len(d)), (len(d),))


def field(nx=24, ny=16, *, slope=(0., 0.), obstacles=None, surface=None, cell=1.):
    x, y = np.meshgrid(np.arange(nx) * cell, np.arange(ny) * cell, indexing="ij")
    if surface is None:
        surface = np.zeros((nx, ny), bool)
        surface[1:-1, 1:-1] = True
    return DistanceField(slope[0] * x + slope[1] * y, surface,
                         np.zeros((nx, ny), bool) if obstacles is None else obstacles,
                         (0, 0), cell, wall_height=2.)


def test_vertical_sloped_ground_holes_near_clipping_and_sky():
    mask = np.zeros((24, 16), bool)
    mask[1:-1, 1:-1] = True
    mask[10:12, 10:12] = False
    scene = GridScene(field(slope=(.1, .2), surface=mask), "cpu")
    down = scene.sensor(rays([[0, 0, -1]]), batch_size=3, near=0, far=20)
    values, valid = down.scan([[3.5, 4.5, 6], [10.5, 10.5, 6], [3.5, 4.5, -2]]).numpy()
    np.testing.assert_array_equal(valid[:, 0], [True, False, False])
    assert values[0, 0] == pytest.approx(6 - .1 * 3.5 - .2 * 4.5, abs=.003)
    up = scene.sensor(rays([[0, 0, 1]]), near=0, far=20)
    assert up.scan([[3.5, 4.5, -2]]).numpy()[0][0, 0] == pytest.approx(3.25, abs=.003)
    assert not up.scan([[3.5, 4.5, 6]]).numpy()[1].any()
    clipped = scene.sensor(rays([[0, 0, -1]]), near=6, far=20)
    assert not clipped.scan([[3.5, 4.5, 6]]).numpy()[1].any()


def test_extruded_obstacles_entry_inside_exit_roof_and_pass_over():
    obstacles = np.zeros((24, 16), bool)
    obstacles[8:12, 3:8] = True
    scene = GridScene(field(obstacles=obstacles), "cpu")
    sensor = scene.sensor(rays([[1, 0, 0], [0, 0, 1], [0, 0, -1]]), near=0, far=40)
    values, valid = sensor.scan([[2, 5, 1]]).numpy()
    np.testing.assert_array_equal(valid[0], [True, False, True])
    np.testing.assert_allclose(values[0, [0, 2]], [5.5, 1], atol=.005)
    values, valid = sensor.scan([[9.5, 5, 1]]).numpy()
    assert valid.all()
    np.testing.assert_allclose(values[0], [2, 1, 1], atol=.005)
    values, valid = sensor.scan([[2, 5, 3]]).numpy()
    np.testing.assert_array_equal(valid[0], [False, False, True])
    assert values[0, 2] == pytest.approx(3, abs=.005)
    down = scene.sensor(rays([[0, 0, -1]]), near=0, far=40)
    assert down.scan([[9.5, 5, 3]]).numpy()[0][0, 0] == pytest.approx(1, abs=.005)
    clipped = scene.sensor(rays([[1, 0, 0]]), near=1, far=40)
    assert not clipped.scan([[7, 5, 1]]).numpy()[1].any()
    backwards = scene.sensor(rays([[-1, 0, 0]]), near=0, far=40)
    assert backwards.scan([[25, 5, 1]]).numpy()[0][0, 0] == pytest.approx(13.5, abs=.005)


def test_bilinear_quadratic_intersection_and_measurement_factor():
    x, y = np.meshgrid(np.arange(-1, 3), np.arange(-1, 3), indexing="ij")
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    scene = GridScene(DistanceField(x * y, mask, np.zeros_like(mask), (-1, -1), 1), "cpu")
    sensor = scene.sensor(rays([[1, 1, 0]]), near=0, far=5, tolerance=1e-6)
    value, valid = sensor.scan([[0, 0, .25]]).numpy()
    assert valid[0, 0]
    assert value[0, 0] == pytest.approx(np.sqrt(.5), abs=5e-5)
    scaled = Rays(sensor.rays.directions, [.5], (1,))
    value, valid = scene.sensor(scaled, near=0, far=5, tolerance=1e-6).scan([[0, 0, .25]]).numpy()
    assert valid[0, 0] and value[0, 0] == pytest.approx(np.sqrt(.5) / 2, abs=5e-5)
    assert sensor.diagnostics()["iteration_limit_events"] == 0


def test_edt_sign_contours_and_conservative_interpolant_gradient():
    mask = np.zeros((24, 16), bool)
    mask[8:12, 3:8] = True
    sdf = _signed_distance(mask, .1)
    assert (sdf[mask] < 0).all() and (sdf[~mask] > 0).all()
    assert sdf[7, 5] + sdf[8, 5] == pytest.approx(0)
    assert _gradient_bound(sdf, .1) <= 1.0001
    # Bilinear gradient is bounded at all interior fractions, not just nodes.
    rng = np.random.default_rng(8)
    for _ in range(100):
        i, j = rng.integers([0, 0], np.array(sdf.shape) - 1)
        x, y = rng.random(2)
        dx = ((1-y)*(sdf[i+1,j]-sdf[i,j]) + y*(sdf[i+1,j+1]-sdf[i,j+1])) / .1
        dy = ((1-x)*(sdf[i,j+1]-sdf[i,j]) + x*(sdf[i+1,j+1]-sdf[i+1,j])) / .1
        assert np.hypot(dx, dy) <= 1.0001
    assert (_signed_distance(np.zeros_like(mask), .1) > 1000).all()


def test_iteration_budget_is_reported_separately_and_resets_without_rebuild():
    mask = np.zeros((24, 16), bool)
    mask[8:12, 3:8] = True
    scene = GridScene(field(obstacles=mask), "cpu")
    sensor = scene.sensor(rays([[1, 0, 0]]), near=0, far=40, max_steps=1)
    assert not sensor.scan([[2, 5, 1]]).numpy()[1].any()
    stats = sensor.diagnostics()
    assert stats["iteration_limit_events"] == 1 and stats["max_iterations_observed"] == 1
    ptr = scene.grid.ptr
    sensor.reset_statistics()
    assert sensor.diagnostics()["iteration_limit_events"] == 0
    assert sensor.diagnostics()["max_iterations_observed"] == 0 and scene.grid.ptr == ptr
    # An upward ray above all geometry is an ordinary miss, not exhaustion.
    sky = scene.sensor(rays([[0, 0, 1]]), near=0, far=40, max_steps=1)
    assert not sky.scan([[2, 5, 3]]).numpy()[1].any()
    assert sky.diagnostics()["iteration_limit_events"] == 0


def test_benchmark_rejects_exhausted_scans():
    from warptracer.benchmark import trial
    from warptracer.execution import ScanRunner
    sim = Simulation(track=OvalTrack(), scenario="drive", engine="lean", device="cpu",
                     lidar=LidarConfig(backend="grid", frequency=60))
    sim.lidar.sensor.max_steps = 1
    runner = ScanRunner(sim, backend="graph", integrator="fused")
    with pytest.raises(RuntimeError, match="exhausted the march budget"):
        trial(runner, "scan", transitions=6, warmup=6, record_hz=30, warmup_wall_seconds=0)


def test_3d_rays_against_independent_sloped_mesh_and_rectangular_obstacle():
    from racesense3d import from_quads
    slope = np.array([.1, -.05])
    obstacles = np.zeros((240, 160), bool)
    obstacles[80:121, 30:81] = True

    def point(x, y, height=0):
        return [x, y, slope @ [x, y] + height]

    floor = [point(.05, .05), point(23.85, .05), point(23.85, 15.85), point(.05, 15.85)]
    bottom = [point(7.95, 2.95), point(12.05, 2.95), point(12.05, 8.05), point(7.95, 8.05)]
    top = [point(7.95, 2.95, 2), point(12.05, 2.95, 2), point(12.05, 8.05, 2), point(7.95, 8.05, 2)]
    quads = [floor, bottom, top]
    for i in range(4):
        j = (i + 1) % 4
        quads.append([bottom[i], bottom[j], top[j], top[i]])
    rng = np.random.default_rng(11)
    directions = rays(rng.normal(size=(100, 3)))
    positions = rng.uniform([-2, -2, -1], [26, 18, 4], size=(25, 3))
    mesh = from_quads(quads, device="cpu").sensor(directions, batch_size=len(positions), near=.01, far=40)
    grid = GridScene(field(nx=240, ny=160, slope=slope, obstacles=obstacles, cell=.1), "cpu").sensor(
        directions, batch_size=len(positions), near=.01, far=40, tolerance=1e-5)
    mv, mm = mesh.scan(positions).numpy()
    gv, gm = grid.scan(positions).numpy()
    np.testing.assert_array_equal(mm, gm)
    # Bilinear EDT contours round raster corners. A grazing ray there can
    # change its first hit; check ordinary intersections separately instead
    # of assuming this representation is an exact rectangular mesh.
    hits = positions[:, None, :] + mv[..., None] * directions.directions
    corners = np.array([[7.95, 2.95], [12.05, 2.95], [12.05, 8.05], [7.95, 8.05],
                        [.05, .05], [23.85, .05], [23.85, 15.85], [.05, 15.85]])
    corner_distance = np.linalg.norm(hits[:, :, None, :2] - corners, axis=-1).min(axis=-1)
    ordinary = mm & (corner_distance > .1)
    assert ordinary.sum() > .98 * mm.sum()
    np.testing.assert_allclose(mv[ordinary], gv[ordinary], atol=.01)
    assert grid.diagnostics()["iteration_limit_events"] == 0


def test_oval_same_pose_comparison_measures_discretization_and_outliers():
    report = oval_scan_comparison(OvalTrack(), "cpu")
    assert report["sensor_poses"] == 432 and report["rays"] == 432 * 108
    assert report["valid_mismatch_fraction"] < .002
    assert report["absolute_range_error_m"]["p99"] < .05
    assert report["common_hits_within_5cm_fraction"] > .99
    assert report["common_hit_errors_over_1m"] > 0
    assert report["absolute_range_error_m"]["max"] > 1
    assert len(report["worst_common_hits"]) == 5
    assert report["candidate_algorithm"] == "edt-sphere-tracing"
    assert report["march_diagnostics"]["iteration_limit_events"] == 0


def test_grid_navigation_completes_laps_without_contacts():
    sim = Simulation(track=OvalTrack(), scenario="drive", engine="lean", device="cpu",
                     lidar=LidarConfig(backend="grid", frequency=60))
    runner = TransitionRunner(sim, controller="disparity", backend="graph", integrator="fused")
    trajectory = sim.run(120, runner=runner)
    a, b = sim.track.center_axes
    phase = np.unwrap(np.arctan2(trajectory.poses[:, 1] / b, trajectory.poses[:, 0] / a))
    assert (phase[-1] - phase[0]) / (2 * np.pi) > 2
    assert sim.wall_contact_substeps.numpy()[0] == 0
    assert np.ptp(trajectory.poses[:, 2]) > .39 and trajectory.lidar.valid.all()
    assert np.linalg.norm(trajectory.velocities[-1, :3]) > .2
    assert trajectory.metadata["lidar"]["march_diagnostics"]["iteration_limit_events"] == 0


def test_short_grid_commands_report_sensor_geometry_and_real_work(tmp_path):
    from warptracer.cli import main
    main(["demo", "oval-grid", "--seconds", ".1", "--output", str(tmp_path)])
    metadata = json.loads((tmp_path / "oval-grid.json").read_text())
    assert metadata["lidar"]["backend"] == "grid"
    assert metadata["lidar"]["grid_cell_size"] == .025
    assert metadata["lidar"]["algorithm"] == "edt-sphere-tracing"
    assert (tmp_path / "oval-grid.html").stat().st_size > 1000
    output = tmp_path / "grid.json"
    main(["benchmark", "grid", "--device", "cpu", "--envs", "1", "--seconds", ".1",
          "--trials", "1", "--warmup-seconds", ".1", "--warmup-wall-seconds", "0", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["lidar_backends"] == ["mesh", "grid"] and report["scan_comparison"]["rays"] == 46656
    assert report["schema_version"] == 4
    assert len(report["results"]) == 4 and len(report["lidar_speedups"]) == 2
    assert all(v["status"] == "passed" for v in report["validation"])
    for result in report["results"]:
        assert result["trials"][0]["wall_contact_substeps"] == [0]
        if result["lidar_backend"] == "grid":
            assert result["lidar_algorithm"] == "edt-sphere-tracing"
            assert result["trials"][0]["march_diagnostics"]["iteration_limit_events"] == 0
        if result["case"] == "scan":
            assert result["physics_substeps_per_transition"] == 0
            assert result["summary"]["median_physics_substeps_per_second"] == 0
            assert result["summary"]["median_simulated_seconds_per_second_per_env"] is None
            assert result["trials"][0]["aggregate_lidar_scans_per_second"] > 0


def test_scan_benchmark_keeps_identical_fixed_poses_for_both_backends():
    from warptracer.benchmark import make_runner
    poses = []
    for backend in ("mesh", "grid"):
        runner = make_runner("scan", "graph", "cpu", 4, num_envs=3, integrator="fused",
                             track="oval", lidar_backend=backend)
        before = runner.sim.snapshot()[0]
        for _ in range(10):
            runner.advance()
        np.testing.assert_array_equal(before, runner.sim.snapshot()[0])
        assert runner.sim.steps == 40
        poses.append(runner.sim.lidar.snapshot()[0])
    np.testing.assert_array_equal(*poses)
    assert not np.allclose(poses[0][0], poses[0][1])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))])
def test_each_grid_batch_entry_matches_independent_car(device):
    def runner(angles):
        sim = Simulation(track=OvalTrack(), scenario="drive", engine="lean", device=device,
                         num_envs=len(angles), lidar=LidarConfig(backend="grid", frequency=40))
        sim.initial_pose[:] = [sim.track.pose(a, sim.drive.ride_height(sim.vehicle)) for a in angles]
        return TransitionRunner(sim, controller="disparity", backend="graph", integrator="fused", substeps=6)
    angles = [-np.pi / 2, 0, np.pi / 2]
    batch = runner(angles)
    single = [runner([angle]) for angle in angles]
    for _ in range(120):
        batch.advance()
        for car in single:
            car.advance()
    poses, velocities = batch.sim.snapshot()
    _, values, valid = batch.sim.lidar.snapshot()
    for i, car in enumerate(single):
        q, qd = car.sim.snapshot()
        _, ranges, mask = car.sim.lidar.snapshot()
        np.testing.assert_allclose(poses[i], q, atol=2e-4)
        np.testing.assert_allclose(velocities[i], qd, atol=2e-4)
        np.testing.assert_allclose(values[i], ranges, atol=2e-4)
        np.testing.assert_array_equal(valid[i], mask)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA driver unavailable")
def test_grid_cuda_scan_agrees_with_cpu_for_tilted_poses():
    data = field(slope=(.1, -.05))
    directions = rays(np.random.default_rng(2).normal(size=(108, 3)))
    poses = [[1.5, 2.5, 3], [15.5, 4.5, -1], [3.5, 8.5, 2]]
    outputs = [GridScene(data, device).sensor(directions, batch_size=3).scan(poses).numpy()
               for device in ("cpu", "cuda:0")]
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=2e-4)


@pytest.mark.parametrize("backend,integrator", [("eager", "fused"), ("graph", "unfused"), ("graph", "fused")])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))])
def test_grid_batched_execution_and_reset_parity(backend, integrator, device):
    def runner(mode, integration):
        sim = Simulation(track=OvalTrack(), scenario="drive", engine="lean", device=device,
                         num_envs=3, lidar=LidarConfig(backend="grid", frequency=60))
        sim.initial_pose[:] = [sim.track.pose(a, sim.drive.ride_height(sim.vehicle))
                              for a in (-np.pi / 2, 0, np.pi / 2)]
        return TransitionRunner(sim, backend=mode, integrator=integration, controller="disparity")
    candidate = runner(backend, integrator)
    assert not hasattr(candidate.sim.lidar.scene, "mesh")
    ptr = candidate.sim.lidar.result.values.ptr
    grid_ptr = candidate.sim.lidar.scene.grid.ptr
    validate_pair(runner("eager", "unfused"), candidate, transitions=120)
    assert candidate.sim.steps == 0 and candidate.sim.lidar.timestamp == 0
    assert candidate.sim.lidar.result.values.ptr == ptr
    assert candidate.sim.lidar.scene.grid.ptr == grid_ptr
    assert candidate.sim.lidar.diagnostics()["iteration_limit_events"] == 0


def test_grid_rejects_invalid_data_and_unsupported_track():
    with pytest.raises(ValueError):
        DistanceField([[0, 0], [0, np.nan]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], (0, 0), 1)
    for bad in (-1, .5, 2):
        mask = np.zeros((4, 4))
        mask[1, 1] = bad
        with pytest.raises(ValueError):
            DistanceField(np.zeros((4, 4)), np.zeros((4, 4)), mask, (0, 0), 1)
    with pytest.raises(ValueError, match="border"):
        DistanceField(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)), (0, 0), 1)
    with pytest.raises(ValueError):
        DistanceField.oval(OvalTrack(), .2)
    with pytest.raises(ValueError, match="oval"):
        Simulation(scenario="drive", engine="lean", lidar=LidarConfig(backend="grid"))
