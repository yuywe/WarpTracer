"""Independent 3D intersections, tile skipping, oval accuracy and capture."""
import json

import numpy as np
import pytest
import warp as wp

from racesense3d.sensors import Rays
from warptracer.benchmark import validate_pair
from warptracer.execution import TransitionRunner
from warptracer.heightfield import GridScene, HeightField
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation
from warptracer.terrain import OvalTrack
from warptracer.lidar_comparison import oval_scan_comparison


def rays(directions):
    d = np.atleast_2d(np.asarray(directions, np.float32))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return Rays(d, np.ones(len(d)), (len(d),))


def field(nx=24, ny=16, *, slope=(0., 0.), obstacles=None, surface=None, tile=8):
    x, y = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    return HeightField(slope[0] * x + slope[1] * y,
                       np.ones((nx, ny)) if surface is None else surface,
                       np.zeros((nx, ny)) if obstacles is None else obstacles, (0, 0), 1, tile)


@pytest.mark.parametrize("tiled", [False, True])
def test_vertical_sloped_ground_holes_near_clipping_and_sky(tiled):
    mask = np.ones((24, 16))
    mask[10, 10] = 0
    scene = GridScene(field(slope=(.1, .2), surface=mask), "cpu", use_tiles=tiled)
    down = scene.sensor(rays([[0, 0, -1]]), batch_size=3, near=0, far=20)
    values, valid = down.scan([[3.5, 4.5, 6], [10.5, 10.5, 6], [3.5, 4.5, -2]]).numpy()
    np.testing.assert_array_equal(valid[:, 0], [True, False, False])
    assert values[0, 0] == pytest.approx(6 - .1 * 3.5 - .2 * 4.5, abs=2e-6)
    up = scene.sensor(rays([[0, 0, 1]]), near=0, far=20)
    assert up.scan([[3.5, 4.5, -2]]).numpy()[0][0, 0] == pytest.approx(3.25, abs=2e-6)
    assert not up.scan([[3.5, 4.5, 6]]).numpy()[1].any()
    clipped = scene.sensor(rays([[0, 0, -1]]), near=6, far=20)
    assert not clipped.scan([[3.5, 4.5, 6]]).numpy()[1].any()


@pytest.mark.parametrize("tiled", [False, True])
def test_extruded_obstacles_entry_inside_exit_roof_and_pass_over(tiled):
    obstacles = np.zeros((24, 16))
    obstacles[8:12, 3:8] = 2
    scene = GridScene(field(obstacles=obstacles), "cpu", use_tiles=tiled)
    sensor = scene.sensor(rays([[1, 0, 0], [0, 0, 1], [0, 0, -1]]), near=0, far=40)
    values, valid = sensor.scan([[2, 5, 1]]).numpy()
    np.testing.assert_array_equal(valid[0], [True, False, True])
    np.testing.assert_allclose(values[0, [0, 2]], [6, 1], atol=1e-5)
    values, valid = sensor.scan([[9.5, 5, 1]]).numpy()
    assert valid.all()
    # Must cross internal occupied-cell faces without a false hit.
    np.testing.assert_allclose(values[0], [2.5, 1, 1], atol=1e-5)
    values, valid = sensor.scan([[2, 5, 3]]).numpy()
    np.testing.assert_array_equal(valid[0], [False, False, True])
    assert values[0, 2] == pytest.approx(3)
    down = scene.sensor(rays([[0, 0, -1]]), near=0, far=40)
    assert down.scan([[9.5, 5, 3]]).numpy()[0][0, 0] == pytest.approx(1)
    # A too-near wall remains the first hit, rather than seeing through it.
    clipped = scene.sensor(rays([[1, 0, 0]]), near=1, far=40)
    assert not clipped.scan([[7.5, 5, 1]]).numpy()[1].any()


def test_bilinear_patch_quadratic_intersection_and_nonunit_measurement_factor():
    h = np.array([[0, 0], [0, 1]], np.float32)  # h(x,y) = xy
    scene = GridScene(HeightField(h, [[1]], [[0]], (0, 0), 1), "cpu")
    sensor = scene.sensor(rays([[1, 1, 0]]), near=0, far=5)
    value, valid = sensor.scan([[0, 0, .25]]).numpy()
    assert valid[0, 0]
    assert value[0, 0] == pytest.approx(np.sqrt(.5), abs=1e-6)
    scaled = Rays(sensor.rays.directions, [.5], (1,))
    value, valid = scene.sensor(scaled, near=0, far=5).scan([[0, 0, .25]]).numpy()
    assert valid[0, 0] and value[0, 0] == pytest.approx(np.sqrt(.5) / 2, abs=1e-6)


def test_boundary_directions_partial_tiles_and_outside_grid():
    obstacles = np.zeros((19, 13))
    obstacles[16:19, 8:13] = 2
    scene = GridScene(field(nx=19, ny=13, obstacles=obstacles), "cpu")
    sensor = scene.sensor(rays([[1, 0, 0], [-1, 0, 0], [0, -1, 0]]), near=0, far=50)
    v, mask = sensor.scan([[16, 10, 1]]).numpy()
    np.testing.assert_allclose(v[0], [0, 0, 2], atol=1e-5)
    assert mask.all()
    v, mask = sensor.scan([[25, 10, 1]]).numpy()
    assert v[0, 1] == pytest.approx(6) and mask[0, 1]
    assert not mask[0, 0]


def test_tiles_match_unaccelerated_traversal_for_random_3d_rays():
    rng = np.random.default_rng(7)
    h = rng.uniform(-.3, .3, (20, 14))
    obstacles = rng.choice([0., 1., 2.], size=(19, 13), p=[.8, .1, .1])
    mask = rng.choice([0, 1], size=(19, 13), p=[.2, .8])
    data = HeightField(h, mask, obstacles, (-2, -3), .25, 8)
    directions = rays(rng.normal(size=(80, 3)))
    positions = rng.uniform([-3, -4, -1], [4, 2, 3], size=(30, 3))
    outputs = []
    for tiled in (False, True):
        sensor = GridScene(data, "cpu", use_tiles=tiled).sensor(directions, batch_size=len(positions), near=.01, far=20)
        outputs.append(sensor.scan(positions).numpy())
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=2e-5)


def test_3d_rays_match_independent_sloped_mesh_and_rectangular_obstacle():
    from racesense3d import from_quads
    slope = np.array([.1, -.05])
    obstacles = np.zeros((24, 16))
    obstacles[8:12, 3:8] = 2

    def point(x, y, height=0):
        return [x, y, slope @ [x, y] + height]

    floor = [point(0, 0), point(24, 0), point(24, 16), point(0, 16)]
    bottom = [point(8, 3), point(12, 3), point(12, 8), point(8, 8)]
    top = [point(8, 3, 2), point(12, 3, 2), point(12, 8, 2), point(8, 8, 2)]
    quads = [floor, bottom, top]
    for i in range(4):
        j = (i + 1) % 4
        quads.append([bottom[i], bottom[j], top[j], top[i]])
    rng = np.random.default_rng(11)
    directions = rays(rng.normal(size=(100, 3)))
    positions = rng.uniform([-2, -2, -1], [26, 18, 4], size=(25, 3))
    outputs = []
    for scene in (from_quads(quads, device="cpu"), GridScene(field(slope=slope, obstacles=obstacles), "cpu")):
        outputs.append(scene.sensor(directions, batch_size=len(positions), near=.01, far=40).scan(positions).numpy())
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=5e-5)


def test_oval_same_pose_comparison_measures_discretization_and_outliers():
    report = oval_scan_comparison(OvalTrack(), "cpu")
    assert report["sensor_poses"] == 432 and report["rays"] == 432 * 108
    assert report["valid_mismatch_fraction"] < .002
    assert report["absolute_range_error_m"]["p99"] < .05
    assert report["common_hits_within_5cm_fraction"] > .99
    assert report["common_hit_errors_over_1m"] > 0
    assert report["absolute_range_error_m"]["max"] > 1
    assert len(report["worst_common_hits"]) == 5


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


def test_short_grid_commands_report_sensor_geometry_and_real_work(tmp_path):
    from warptracer.cli import main
    main(["demo", "oval-grid", "--seconds", ".1", "--output", str(tmp_path)])
    metadata = json.loads((tmp_path / "oval-grid.json").read_text())
    assert metadata["lidar"]["backend"] == "grid"
    assert metadata["lidar"]["grid_cell_size"] == .025
    assert (tmp_path / "oval-grid.html").stat().st_size > 1000
    output = tmp_path / "grid.json"
    main(["benchmark", "grid", "--device", "cpu", "--envs", "1", "--seconds", ".1",
          "--trials", "1", "--warmup-seconds", ".1", "--warmup-wall-seconds", "0", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["lidar_backends"] == ["mesh", "grid"] and report["scan_comparison"]["rays"] == 46656
    assert len(report["results"]) == 4 and len(report["lidar_speedups"]) == 2
    assert all(v["status"] == "passed" for v in report["validation"])
    for result in report["results"]:
        assert result["trials"][0]["wall_contact_substeps"] == [0]
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
    validate_pair(runner("eager", "unfused"), candidate, transitions=120)
    assert candidate.sim.steps == 0 and candidate.sim.lidar.timestamp == 0
    assert candidate.sim.lidar.result.values.ptr == ptr


def test_grid_rejects_invalid_data_and_unsupported_track():
    with pytest.raises(ValueError):
        HeightField([[0, 0], [0, np.nan]], [[1]], [[0]], (0, 0), 1)
    with pytest.raises(ValueError):
        HeightField([[0, 0], [0, 0]], [[1]], [[-1]], (0, 0), 1)
    with pytest.raises(ValueError):
        HeightField([[0, 0], [0, 0]], [[.5]], [[0]], (0, 0), 1)
    with pytest.raises(ValueError):
        HeightField.oval(OvalTrack(), .2)
    with pytest.raises(ValueError, match="oval"):
        Simulation(scenario="drive", engine="lean", lidar=LidarConfig(backend="grid"))
