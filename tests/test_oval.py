"""Elevation, full-loop navigation, footprint containment and captured parity."""
import json

import numpy as np
import pytest
import warp as wp

from racesense3d import lidar
from racesense3d.core import Scene
from warptracer.benchmark import validate_pair
from warptracer.cli import main
from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation
from warptracer.terrain import OvalTrack, road_rotation

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))]


def car(device="cpu", n=1, hz=60, **kwargs):
    return Simulation(track=OvalTrack(), scenario="drive", engine="lean", device=device,
                      num_envs=n, lidar=LidarConfig(frequency=hz), **kwargs)


def rotate(q, points):
    t = 2 * np.cross(q[:3], points)
    return points + q[3] * t + np.cross(q[:3], t)


def test_mesh_is_closed_at_seam_and_ground_matches_surface():
    track = OvalTrack()
    parts = track.mesh_parts()
    assert set(parts) == {"road", "inner_barrier", "outer_barrier"}
    for name, (vertices, faces) in parts.items():
        tri = vertices[faces]
        assert np.all(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 0)
        if name != "road":
            edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            np.testing.assert_array_equal(counts, 2)
    angles = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    a, b = track.center_axes
    x, y = a * np.cos(angles), b * np.sin(angles)
    origins = np.column_stack((x, y, track.height(x) + 1))
    scene = Scene(*track.mesh(), device="cpu")
    sensor = scene.sensor(lidar([0]), batch_size=len(origins), near=0, far=2)
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32)
    result = sensor.scan(origins, rotation)
    ranges, valid = result.numpy()
    assert valid.all()
    np.testing.assert_allclose(ranges, 1, atol=.001)


@pytest.mark.parametrize("hz", [30, 40, 60])
def test_oval_navigation_completes_laps_with_height_and_tilt(hz):
    sim = car(hz=hz)
    runner = TransitionRunner(sim, controller="disparity", backend="graph", integrator="fused", substeps=240 // hz)
    trajectory = sim.run(120, record_fps=20 if hz == 40 else 30, runner=runner)
    q = trajectory.poses
    a, b = sim.track.center_axes
    phase = np.unwrap(np.arctan2(q[:, 1] / b, q[:, 0] / a))
    assert (phase[-1] - phase[0]) / (2 * np.pi) > 2
    assert np.ptp(q[:, 2]) > .39
    assert np.linalg.norm(trajectory.velocities[-1, :3]) > .2
    assert sim.wall_contact_substeps.numpy()[0] == 0
    gradient = sim.track.gradient(q[:, 0])
    expected_z = sim.track.height(q[:, 0]) + sim.drive.ride_height(sim.vehicle) * np.sqrt(1 + gradient**2)
    np.testing.assert_allclose(q[:, 2], expected_z, atol=1e-6)
    normal = np.array([rotate(p[3:], np.array([0., 0, 1])) for p in q])
    expected_normal = np.column_stack((-gradient, np.zeros_like(gradient), np.ones_like(gradient)))
    expected_normal /= np.linalg.norm(expected_normal, axis=1, keepdims=True)
    np.testing.assert_allclose(normal, expected_normal, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(q[:, 3:], axis=1), 1, atol=1e-6)
    assert trajectory.lidar.valid.all()
    # Mounted scanner inherits the body tilt, rather than staying world-horizontal.
    sample = 1
    sensor = trajectory.lidar.poses[sample]
    runner.reset()
    runner.advance()
    body, _ = sim.snapshot()
    np.testing.assert_allclose(sensor[:3], body[:3] + rotate(body[3:], np.array(sim.lidar.mount_position)), atol=1e-6)
    np.testing.assert_allclose(sensor[3:], body[3:], atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend,integrator", [("eager", "unfused"), ("graph", "unfused"), ("eager", "fused"), ("graph", "fused")])
def test_distinct_tilted_spawns_and_reset_match_reference(device, backend, integrator):
    def runner(mode, integration):
        sim = car(device, n=3)
        sim.initial_pose[:] = [sim.track.pose(angle, sim.drive.ride_height(sim.vehicle))
                               for angle in (-np.pi / 2, 0, np.pi / 2)]
        return TransitionRunner(sim, controller="disparity", backend=mode, integrator=integration)
    reference, candidate = runner("eager", "unfused"), runner(backend, integrator)
    before = candidate.sim.snapshot()[0]
    validate_pair(reference, candidate, transitions=720)
    np.testing.assert_allclose(candidate.sim.snapshot()[0], before, atol=1e-6)
    assert candidate.sim.steps == 0 and candidate.sim.lidar.timestamp == 0
    for _ in range(720):
        candidate.advance()
    assert candidate.sim.wall_contact_substeps.numpy().sum() == 0
    assert np.isfinite(candidate.sim.snapshot()[0]).all()


@pytest.mark.parametrize("wall", ["outer", "inner"])
def test_oval_barriers_stop_the_whole_chassis(wall):
    sim = car()
    if wall == "inner":
        sim.initial_pose[0, 1] = -3.7
        sim.initial_pose[0, 3:] = road_rotation(float(np.pi / 2), float(sim.track.gradient(0)))
        sim.reset()
    sim.set_target_speed(3)
    half = np.array(sim.vehicle.dimensions) / 2
    corners = np.array([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    for _ in range(960):
        sim.step()
        pose, velocity = sim.snapshot()
        world = rotate(pose[3:], corners) + pose[:3]
        assert np.max((world[:, 0] / 8)**2 + (world[:, 1] / 5)**2) <= 1 + 1e-6
        points = world[:, :2] / [6, 3]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                segment = points[j] - points[i]
                if np.dot(segment, segment) > 1e-12:
                    t = np.clip(-np.dot(points[i], segment) / np.dot(segment, segment), 0, 1)
                    assert np.linalg.norm(points[i] + t * segment) >= 1 - 1e-6
    assert sim.wall_contact_substeps.numpy()[0] > 0
    np.testing.assert_array_equal(velocity, 0)
    sim.reset()
    assert sim.wall_contact_substeps.numpy()[0] == 0


def test_tilted_spawn_heading_and_vertical_velocity_are_preserved():
    sim = car()
    sim.initial_pose[0] = sim.track.pose(np.pi / 2, sim.drive.ride_height(sim.vehicle))
    sim.reset()
    sim.set_target_speed(1)
    for _ in range(240):
        before = sim.snapshot()[0]
        sim.step()
        after, velocity = sim.snapshot()
        np.testing.assert_allclose(velocity[2], (after[2] - before[2]) / sim.dt, atol=1e-5)
    assert after[0] < -.5 and velocity[0] < -.5 and velocity[2] < 0
    assert sim.wall_contact_substeps.numpy()[0] == 0


def test_short_oval_command_exports_track_and_lap_metadata(tmp_path):
    main(["demo", "oval", "--seconds", ".2", "--output", str(tmp_path)])
    report = json.loads((tmp_path / "oval.json").read_text())
    assert report["track_kind"] == "oval" and report["track"]["elevation"] == .4
    assert report["completed_laps"] == 0 and report["wall_contact_substeps"] == 0
    assert (tmp_path / "oval.html").stat().st_size > 1000


def test_oval_benchmark_preset_measures_navigation_on_the_loop(tmp_path):
    path = tmp_path / "oval-benchmark.json"
    main(["benchmark", "oval", "--device", "cpu", "--envs", "3", "--seconds", ".1",
          "--trials", "1", "--warmup-seconds", ".1", "--warmup-wall-seconds", "0", "--output", str(path)])
    report = json.loads(path.read_text())
    assert report["preset"] == "oval" and report["track_kind"] == "oval"
    assert report["track"]["elevation"] == .4
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result["controller"] == "disparity" and result["environments"] == 3
    assert result["trials"][0]["wall_contact_substeps"] == [0, 0, 0]


def test_oval_benchmark_does_not_overwrite_demo_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    demo_metadata = tmp_path / "outputs/oval.json"
    demo_metadata.write_text('{"controller": "disparity", "completed_laps": 4}')
    main(["benchmark", "oval", "--device", "cpu", "--envs", "1", "--seconds", ".1",
          "--trials", "1", "--warmup-seconds", ".1", "--warmup-wall-seconds", "0"])
    report = json.loads((tmp_path / "outputs/oval-benchmark.json").read_text())
    assert report["track_kind"] == "oval"
    assert json.loads(demo_metadata.read_text())["completed_laps"] == 4


def test_oval_rejects_invalid_geometry_and_unsupported_physics():
    with pytest.raises(ValueError, match="grade"):
        OvalTrack(elevation=5)
    with pytest.raises(ValueError, match="segments"):
        OvalTrack(segments=31)
    with pytest.raises(ValueError, match="lane_width"):
        OvalTrack(lane_width=6)
    with pytest.raises(ValueError, match="thickness"):
        OvalTrack(barrier_thickness=3)
    with pytest.raises(ValueError, match="lean"):
        Simulation(track=OvalTrack(), scenario="drive", engine="newton", device="cpu")
