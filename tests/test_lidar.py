"""Mounted scan checks against analytic geometry, independent of mesh ray casting."""
import numpy as np
import pytest
import warp as wp

from warptracer.lidar import LidarConfig
from warptracer.scene import Track
from warptracer.simulation import Simulation


def rotation_matrix(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def reference_scan(track, origin, directions, near, far):
    """Vectorized slab intersections for boxes, plus analytic infinite floor."""
    distance = np.full(len(directions), np.inf)
    for center, size in track.barriers():
        lo, hi = np.asarray(center)-np.asarray(size)/2, np.asarray(center)+np.asarray(size)/2
        enter, leave = np.full(len(directions), -np.inf), np.full(len(directions), np.inf)
        for axis in range(3):
            parallel = np.abs(directions[:, axis]) < 1e-10
            a = np.divide(lo[axis]-origin[axis], directions[:, axis],
                          out=np.full(len(directions), -np.inf), where=~parallel)
            b = np.divide(hi[axis]-origin[axis], directions[:, axis],
                          out=np.full(len(directions), np.inf), where=~parallel)
            enter = np.maximum(enter, np.minimum(a, b))
            leave = np.minimum(leave, np.maximum(a, b))
            if not lo[axis] <= origin[axis] <= hi[axis]:
                leave[parallel] = -np.inf
        hit = np.where(enter >= 0, enter, leave)
        distance = np.minimum(distance, np.where(leave >= np.maximum(enter, 0), hit, np.inf))
    floor = np.divide(-origin[2], directions[:, 2], out=np.full(len(directions), np.inf),
                      where=np.abs(directions[:, 2]) > 1e-10)
    distance = np.minimum(distance, np.where(floor >= 0, floor, np.inf))
    valid = (distance >= near) & (distance < far)
    return np.where(valid, distance, far), valid


def test_known_wall_ranges_and_rotated_mount_offset():
    sim = Simulation(track=Track(), scenario="drive", device="cpu",
                     lidar=LidarConfig(beams=3, fov_degrees=180))
    pose, values, valid = sim.lidar.snapshot()
    np.testing.assert_allclose(values, [2, 2.88, 2], atol=1e-5)
    assert valid.all()
    np.testing.assert_allclose(pose[:2], [.12, 0], atol=1e-6)
    q = sim.state.body_q.numpy()
    q[0, :2] = [1, .3]
    q[0, 3:] = [0, 0, np.sqrt(.5), np.sqrt(.5)]
    sim.state.body_q.assign(q)
    sim.lidar.update(sim.state, sim.body, 0)
    pose, values, valid = sim.lidar.snapshot()
    np.testing.assert_allclose(pose[:2], [1, .42], atol=1e-6)
    np.testing.assert_allclose(values, [2, 1.58, 4], atol=1e-5)
    assert valid.all()


@pytest.mark.parametrize("pitch,near,hit", [(np.pi/2, .021, True),
                                           (np.pi/2, .5, False), (-np.pi/2, .021, False)])
def test_mount_pitch_floor_and_invalid_returns(pitch, near, hit):
    sim = Simulation(scenario="drive", device="cpu",
                     lidar=LidarConfig(beams=1, mount_rpy=(0, pitch, 0), near=near))
    pose, values, valid = sim.lidar.snapshot()
    assert valid[0] == hit
    assert values[0] == pytest.approx(pose[2] if hit else sim.lidar.config.far, abs=1e-5)


def test_full_body_and_mount_rotation_composition():
    config = LidarConfig(beams=9, mount_position=(.1, .02, .08), mount_rpy=(.1, .2, -.3))
    sim = Simulation(scenario="drive", device="cpu", lidar=config)
    q = sim.state.body_q.numpy()
    q[0] = [.2, -.3, .5, *wp.quat_rpy(.2, -.15, .6)]
    sim.state.body_q.assign(q)
    sim.lidar.update(sim.state, sim.body, .25)
    pose, values, valid = sim.lidar.snapshot()
    body_R = rotation_matrix(q[0, 3:])
    expected_R = body_R @ rotation_matrix(np.array(wp.quat_rpy(*config.mount_rpy)))
    np.testing.assert_allclose(pose[:3], q[0, :3] + body_R @ config.mount_position, atol=1e-6)
    np.testing.assert_allclose(rotation_matrix(pose[3:]), expected_R, atol=1e-6)
    reference, mask = reference_scan(sim.track, pose[:3], sim.lidar.rays.directions @ expected_R.T,
                                      config.near, config.far)
    np.testing.assert_allclose(values, reference, atol=1e-4)
    np.testing.assert_array_equal(valid, mask)


@pytest.mark.parametrize("scenario", ["accelerate-brake", "circle", "s-turn"])
@pytest.mark.parametrize("beams", [108, 1080])
def test_moving_scans_match_analytic_geometry(scenario, beams):
    sim = Simulation(scenario=scenario, device="cpu", lidar=LidarConfig(beams=beams))
    tr = sim.run(3)
    scans = tr.lidar
    np.testing.assert_array_equal(scans.times, tr.times)
    assert scans.ranges.shape == (91, beams)
    for i in (0, 20, 50, 90):
        body_R = rotation_matrix(tr.poses[i, 3:])
        np.testing.assert_allclose(scans.poses[i, :3],
                                   tr.poses[i, :3] + body_R @ sim.lidar.mount_position, atol=1e-6)
        directions = scans.directions @ rotation_matrix(scans.poses[i, 3:]).T
        expected, valid = reference_scan(sim.track, scans.poses[i, :3], directions,
                                         sim.lidar.config.near, sim.lidar.config.far)
        np.testing.assert_array_equal(scans.valid[i], valid)
        np.testing.assert_allclose(scans.ranges[i], expected, atol=1e-4, rtol=1e-5)
        points = scans.points(i)
        np.testing.assert_allclose(points[valid],
                                   scans.poses[i, :3] + directions[valid] * expected[valid, None], atol=1e-4)
    assert not np.allclose(scans.ranges[0], scans.ranges[-1])


def test_scan_cadence_independent_of_recording_and_headless():
    sim = Simulation(scenario="drive", device="cpu", lidar=LidarConfig(frequency=20))
    tr = sim.run(.13, record_fps=30)
    np.testing.assert_allclose(tr.lidar.times, [0, .05, .1])
    assert len(tr.times) == 5 and tr.times[-1] > tr.lidar.times[-1]
    headless = sim.run(.13, record=False)
    np.testing.assert_allclose(headless.lidar.times, [0, .1])
    np.testing.assert_array_equal(headless.lidar.ranges[-1], tr.lidar.ranges[-1])
    sim.reset()
    assert sim.lidar.timestamp == 0
    ptr = sim.lidar.result.values.ptr
    for _ in range(12):
        sim.step()
    assert sim.lidar.timestamp == .05 and sim.lidar.result.values.ptr == ptr


def test_sensing_does_not_change_vehicle_motion():
    outputs = []
    for config in (None, LidarConfig()):
        sim = Simulation(scenario="circle", device="cpu", lidar=config)
        outputs.append(sim.run(1))
        assert sim.model.body_count == 1 and sim.model.shape_count == 6
    np.testing.assert_array_equal(outputs[0].poses, outputs[1].poses)
    np.testing.assert_array_equal(outputs[0].velocities, outputs[1].velocities)


def test_invalid_lidar_configuration_and_full_circle():
    for kwargs in ({"beams": 0}, {"frequency": 0}, {"far": .001}, {"mount_rpy": (0, np.nan, 0)}):
        with pytest.raises(ValueError):
            LidarConfig(**kwargs)
    with pytest.raises(ValueError):
        Simulation(scenario="drive", device="cpu", lidar=LidarConfig(frequency=31))
    rays = LidarConfig(beams=360, fov_degrees=360).rays()
    assert not np.allclose(rays.directions[0], rays.directions[-1])


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA driver unavailable")
def test_cuda_mounted_scan_agrees_with_cpu():
    values = []
    for device in ("cpu", "cuda:0"):
        sim = Simulation(scenario="drive", device=device, lidar=LidarConfig())
        values.append(sim.lidar.snapshot())
    np.testing.assert_array_equal(values[0][2], values[1][2])
    np.testing.assert_allclose(values[0][1], values[1][1], atol=1e-4)
