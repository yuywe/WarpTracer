"""Observable driving behavior of the flat-ground backend."""
import numpy as np
import pytest

from warptracer.scene import Track, Vehicle
from warptracer.simulation import Simulation


def car(**kwargs):
    return Simulation(scenario="drive", engine="lean", device="cpu", **kwargs)


def test_acceleration_braking_and_fixed_height():
    sim = car()
    tr = sim.run(3, controller=lambda s, t: s.set_action(
        throttle=float(t < 1), brake=float(t >= 1)))
    assert 1.6 < tr.velocities[:, 0].max() < 2.1
    assert tr.velocities[:, 0].min() > -1e-5
    assert abs(tr.velocities[-1, 0]) < .02
    np.testing.assert_allclose(tr.poses[:, 2], tr.poses[0, 2], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(tr.poses[:, 3:], axis=1), 1, atol=1e-6)


def test_stationary_steering_and_reset():
    sim = car()
    initial = sim.snapshot()[0]
    tr = sim.run(.5, controller=lambda s, t: s.set_action(steering=.3))
    np.testing.assert_allclose(tr.poses[-1], initial, atol=1e-6)
    sim.set_target_speed(1, .3)
    for _ in range(120):
        sim.step()
    assert np.linalg.norm(sim.snapshot()[1]) > .3
    sim.reset()
    np.testing.assert_array_equal(sim.snapshot()[0], initial)
    np.testing.assert_array_equal(sim.lean_motion.numpy(), 0)
    np.testing.assert_array_equal(sim.applied_controls.numpy(), 0)


def test_turns_are_symmetric():
    trajectories = []
    for direction in (-1, 1):
        sim = car()
        tr = sim.run(2, controller=lambda s, t: s.set_target_speed(1, direction * .25))
        assert direction * tr.poses[-1, 1] > .3
        assert direction * tr.velocities[-1, 5] > .4
        assert np.max(np.abs(np.diff(tr.actions[:, 2]))) <= sim.drive.steering_rate / 30 + 1e-5
        trajectories.append(tr)
    np.testing.assert_allclose(trajectories[0].poses[:, 0], trajectories[1].poses[:, 0], atol=1e-5)
    np.testing.assert_allclose(trajectories[0].poses[:, 1], -trajectories[1].poses[:, 1], atol=1e-5)


def test_zero_grip_cannot_accelerate_or_turn_a_coasting_car():
    sim = car(vehicle=Vehicle(friction=0))
    sim.lean_motion.assign(np.array([[1, 0, 0, 0]], dtype=np.float32))
    sim.set_action(throttle=1, steering=.3)
    for _ in range(120):
        sim.step()
    q, qd = sim.snapshot()
    assert .8 < qd[0] < 1
    np.testing.assert_allclose(qd[1:], 0, atol=1e-6)
    np.testing.assert_allclose(q[3:], [0, 0, 0, 1], atol=1e-6)


def test_wall_stops_chassis_and_keeps_rotated_footprint_inside():
    sim = car(track=Track(length=4, width=4))
    sim.set_target_speed(3, .2)
    collided = False
    for _ in range(960):
        sim.step()
        q, qd = sim.snapshot()
        yaw = 2 * np.arctan2(q[5], q[6])
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        assert abs(q[0]) + .5 * (c * sim.vehicle.length + s * sim.vehicle.width) <= 2 + 1e-5
        assert abs(q[1]) + .5 * (s * sim.vehicle.length + c * sim.vehicle.width) <= 2 + 1e-5
        if sim.collision.numpy()[0]:
            collided = True
            np.testing.assert_array_equal(qd, 0)
    assert collided


def test_lean_rejects_free_body_scenarios():
    with pytest.raises(ValueError):
        Simulation(scenario="drop", engine="lean", device="cpu")
