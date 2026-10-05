"""Observable scan reactions and closed-loop navigation across execution modes."""
import numpy as np
import pytest
import warp as wp

from warptracer.disparity import DisparityConfig, DisparityController, extend_disparities
from warptracer.execution import TransitionRunner
from warptracer.lidar import LidarConfig
from warptracer.simulation import Simulation

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))]


def car(n=1, device="cpu", hz=60):
    return Simulation(scenario="drive", engine="lean", device=device, num_envs=n,
                      lidar=LidarConfig(frequency=hz))


@pytest.mark.parametrize("device", DEVICES)
def test_extension_covers_far_side_without_overwriting_closer_obstacles(device):
    # At 1 m, radius .28 covers one .3-radian sample on either side.
    raw = wp.array([[5, 5, 1, 5, 5, 5, 5], [5, 1, 5, .5, 5, 5, 5]], dtype=float, device=device)
    valid = wp.ones((2, 7), dtype=int, device=device)
    out = wp.empty_like(raw)
    wp.launch(extend_disparities, dim=(2, 7), inputs=[raw, valid, out, 7, .3, .28, .5, 6.], device=device)
    np.testing.assert_allclose(out.numpy()[0], [5, 1, 1, 1, 5, 5, 5])
    # The nearer .5 m obstacle extends two rays; overlapping extensions use min.
    np.testing.assert_allclose(out.numpy()[1], [1, .5, .5, .5, .5, .5, 5])
    np.testing.assert_array_equal(raw.numpy()[0], [5, 5, 1, 5, 5, 5, 5])


@pytest.mark.parametrize("device", DEVICES)
def test_scan_controls_turn_direction_and_invalid_scans_stop(device):
    sim = car(4, device)
    controller = DisparityController(sim)
    values = np.full((4, 108), 4., np.float32)
    angles = np.linspace(-135, 135, 108)
    values[0, angles > 0] = .7  # left blocked -> right
    values[1] = values[0, ::-1]  # right blocked -> left
    values[2] = .2  # too close to proceed
    values[3] = np.nan
    sim.lidar.result.values.assign(values)
    sim.lidar.result.valid.assign(np.ones_like(values, dtype=np.int32))
    controller.update()
    commands = sim.device_commands.numpy()
    assert commands[0, 2] < 0 < commands[1, 2]
    assert commands[0, 3] > 0 and commands[1, 3] > 0
    np.testing.assert_allclose(commands[0, 2], -commands[1, 2], atol=1e-6)
    np.testing.assert_array_equal(commands[2:, 3], 0)
    assert np.isfinite(commands).all()
    sim.lidar.result.valid.zero_()
    controller.update()
    np.testing.assert_array_equal(sim.device_commands.numpy()[:, 3], 0)


@pytest.mark.parametrize("hz", [30, 40, 60])
def test_navigation_moves_and_turns_without_wall_contacts(hz):
    sim = car(hz=hz)
    r = TransitionRunner(sim, backend="graph", integrator="fused", substeps=240 // hz, controller="disparity")
    positions = []
    for _ in range(60 * hz):
        r.advance()
        positions.append(sim.snapshot()[0][:2])
    positions = np.asarray(positions)
    distance = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    assert distance > 20  # excludes a controller that simply stops to avoid walls
    assert np.ptp(positions[:, 0]) > 3 and np.ptp(positions[:, 1]) > 2
    assert sim.wall_contact_substeps.numpy()[0] == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend,integrator", [("eager", "unfused"), ("graph", "unfused"), ("eager", "fused"), ("graph", "fused")])
def test_distinct_batch_matches_independent_cars_and_reset(device, backend, integrator):
    sim = car(3, device)
    sim.initial_pose[:, :2] = [[-1, -.8], [1, .6], [0, 1.5]]
    batch = TransitionRunner(sim, backend=backend, integrator=integrator, controller="disparity")
    singles = []
    for i in range(3):
        s = car(device=device)
        s.initial_pose[:] = sim.initial_pose[i]
        singles.append(TransitionRunner(s, controller="disparity"))
    for _ in range(2):
        for r in [batch, *singles]:
            r.reset()
        for _ in range(600):
            for r in [batch, *singles]:
                r.advance()
        q, v = sim.snapshot()
        for i, single in enumerate(singles):
            sq, sv = single.sim.snapshot()
            np.testing.assert_allclose(q[i], sq, atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(v[i], sv, atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(sim.device_commands.numpy()[i], single.sim.device_commands.numpy()[0], atol=2e-4)
        assert not np.allclose(q[0], q[1])
        assert sim.steps == 2400


def test_brakes_before_front_wall_and_contact_counter_resets():
    sim = car()
    sim.initial_pose[0, 0] = 5.3
    r = TransitionRunner(sim, backend="graph", integrator="fused", controller="disparity")
    sim.lean_motion.assign(np.array([[1, 0, 0, 0]], np.float32))
    r.advance()
    assert sim.applied_controls.numpy()[0, 1] > 0
    for _ in range(120):
        r.advance()
    assert sim.wall_contact_substeps.numpy()[0] == 0
    sim.wall_contact_substeps.assign(np.array([10], np.int32))
    r.reset()
    assert sim.wall_contact_substeps.numpy()[0] == 0


def test_unsupported_sensor_and_parameters_rejected():
    with pytest.raises(ValueError, match="LiDAR"):
        DisparityController(Simulation(scenario="drive", engine="lean", device="cpu"))
    with pytest.raises(ValueError, match="unrotated"):
        DisparityController(Simulation(scenario="drive", engine="lean", device="cpu",
                                       lidar=LidarConfig(mount_rpy=(0, 0, .2))))
    with pytest.raises(ValueError):
        DisparityConfig(max_speed=float("nan"))
    with pytest.raises(ValueError, match="search sector"):
        DisparityController(car(), DisparityConfig(search_degrees=.5))


@pytest.mark.parametrize("device", DEVICES)
def test_corner_override_checks_its_own_filtered_clearance(device):
    sim = car(device=device)
    controller = DisparityController(sim)
    angles = np.linspace(-135, 135, 108)
    values = np.full((1, 108), 6., np.float32)
    values[0, np.abs(angles) < 10] = 1.5
    values[0, (angles < -30) & (angles > -90)] = .8
    values[0, (angles > 65) & (angles < 90)] = .4
    sim.lidar.result.values.assign(values)
    sim.lidar.result.valid.assign(np.ones_like(values, dtype=np.int32))
    controller.update()
    angle, distance = controller.goals.numpy()[0]
    ray = np.argmin(np.abs(np.deg2rad(angles) - angle))
    assert controller.turns.numpy()[0] == 1
    np.testing.assert_allclose(angle, np.deg2rad(angles[ray]), atol=1e-6)
    np.testing.assert_allclose(distance, controller.filtered.numpy()[0, ray])
    assert distance <= .4 + 1e-6
    assert sim.device_commands.numpy()[0, 3] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_low_range_cap_does_not_latch_turn_in_clear_space(device):
    sim = car(device=device)
    controller = DisparityController(sim, DisparityConfig(range_cap=3.))
    sim.lidar.result.valid.assign(np.ones((1, 108), np.int32))
    sim.lidar.result.values.assign(np.full((1, 108), .8, np.float32))
    controller.update()
    assert controller.turns.numpy()[0] != 0
    sim.lidar.result.values.assign(np.full((1, 108), 20., np.float32))
    controller.update()
    assert controller.turns.numpy()[0] == 0
    assert abs(sim.device_commands.numpy()[0, 2]) < .03
    assert sim.device_commands.numpy()[0, 3] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_direct_controller_update_allows_manual_override(device):
    sim = car(device=device)
    controller = DisparityController(sim)
    controller.update()
    assert sim.device_commands.numpy()[0, 3] > 0
    sim.set_action(throttle=0, brake=0, steering=0)
    np.testing.assert_array_equal(sim.device_commands.numpy()[0], [0, 0, 0, -1])


def test_transition_recording_keeps_sensor_and_pose_timestamps():
    sim = car(hz=40)
    r = TransitionRunner(sim, controller="disparity", backend="graph", integrator="fused", substeps=6)
    tr = sim.run(1, record_fps=20, runner=r)
    np.testing.assert_allclose(tr.times, np.arange(21) / 20)
    np.testing.assert_allclose(tr.lidar.times, np.arange(41) / 40)
    assert tr.actions.shape == (21, 3)
    assert tr.lidar.ranges.shape == (41, 108)
    assert tr.metadata["controller"] == "disparity"
    assert tr.metadata["wall_contact_substeps"] == 0
    assert tr.poses[-1, 0] > .1
    headless = sim.run(1, record=False, runner=r)
    np.testing.assert_allclose(headless.poses[-1], tr.poses[-1], atol=1e-6)
