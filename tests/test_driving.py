"""Behavior checks for the force-based box vehicle, independent of the viewer."""
import numpy as np
import pytest
import warp as wp

from warptracer.driving import DriveConfig
from warptracer.scene import Track, Vehicle
from warptracer.simulation import Simulation


def test_neutral_support_and_reset():
    sim = Simulation(scenario="drive", device="cpu")
    initial, _ = sim.snapshot()
    tr = sim.run(1)
    np.testing.assert_allclose(tr.poses[-1], initial, atol=1e-4)
    assert np.linalg.norm(tr.velocities[-1]) < 1e-3
    sim.set_action(throttle=1, steering=.2)
    sim.step()
    sim.reset()
    np.testing.assert_allclose(sim.snapshot()[0], initial, atol=1e-6)
    np.testing.assert_array_equal(sim.applied_controls.numpy(), 0)
    assert sim.command == (0, 0, 0) and sim.target_speed == -1


def test_throttle_accelerates_and_brake_stops_without_reversing():
    sim = Simulation(scenario="drive", device="cpu")
    def driver(sim, t):
        sim.set_action(throttle=1 if t < 1 else 0, brake=1 if t >= 1 else 0)
    tr = sim.run(3, controller=driver)
    speed = tr.velocities[:, 0]
    assert 1.6 < speed.max() < 2.1
    assert np.min(speed) > -.01
    assert abs(speed[-1]) < .02
    assert np.max(np.abs(tr.poses[:, 1])) < 1e-4
    # The car travels less than one meter after braking begins.
    braking = tr.times >= 1
    assert tr.poses[-1, 0] - tr.poses[braking, 0][0] < 1


def test_coasting_retains_more_speed_than_braking():
    final = []
    for braking in (False, True):
        sim = Simulation(scenario="drive", device="cpu")
        def driver(sim, t):
            sim.set_action(throttle=float(t < .5), brake=float(braking and t >= .5))
        tr = sim.run(1.5, controller=driver)
        final.append(tr.velocities[-1, 0])
    assert final[0] > .6 and abs(final[1]) < .03


def test_steering_turns_both_directions_symmetrically():
    results = []
    for direction in (-1, 1):
        sim = Simulation(scenario="drive", device="cpu")
        tr = sim.run(2, controller=lambda sim, t: sim.set_target_speed(1, direction * .25))
        assert direction * tr.poses[-1, 1] > .3
        assert direction * tr.velocities[-1, 5] > .4
        assert np.max(np.abs(np.diff(tr.actions[:, 2]))) <= sim.drive.steering_rate / 30 + 1e-5
        results.append(tr)
    np.testing.assert_allclose(results[0].poses[:, 0], results[1].poses[:, 0], atol=.002)
    np.testing.assert_allclose(results[0].poses[:, 1], -results[1].poses[:, 1], atol=.002)


def test_no_grip_means_no_propulsion():
    sim = Simulation(scenario="drive", vehicle=Vehicle(friction=0), device="cpu")
    tr = sim.run(.5, controller=lambda sim, t: sim.set_action(throttle=1, steering=.3))
    assert np.max(np.linalg.norm(tr.velocities[:, :2], axis=1)) < 1e-5


def test_airborne_tires_cannot_accelerate_or_steer():
    sim = Simulation(scenario="drive", device="cpu")
    pose = sim.state.body_q.numpy()
    pose[0, 2] = 2.0
    sim.state.body_q.assign(pose)
    sim.set_action(throttle=1, steering=.3)
    for _ in range(24):
        sim.step()
    q, velocity = sim.snapshot()
    assert q[2] > 1.9
    np.testing.assert_allclose(velocity[:2], 0, atol=1e-6)
    np.testing.assert_allclose(velocity[3:], 0, atol=1e-6)
    assert velocity[2] == pytest.approx(-.981, abs=.002)


def test_controls_clamp_and_reject_invalid_values():
    sim = Simulation(scenario="drive", device="cpu")
    sim.set_action(throttle=2, brake=2, steering=2)
    sim.step()
    applied = sim.applied_controls.numpy()[0]
    assert applied[0] == 0 and applied[1] == 1  # braking wins
    assert applied[2] <= sim.drive.steering_rate * sim.dt + 1e-6
    with pytest.raises(ValueError):
        sim.set_action(throttle=np.nan)
    with pytest.raises(ValueError):
        sim.set_target_speed(-1)
    with pytest.raises(ValueError):
        DriveConfig(wheelbase=0)
    with pytest.raises(ValueError):
        Simulation(scenario="drive", physics_hz=60, device="cpu")


def test_driven_car_stops_at_wall_under_continued_throttle():
    sim = Simulation(scenario="drive", track=Track(length=4, width=4), device="cpu")
    tr = sim.run(4, controller=lambda sim, t: sim.set_target_speed(2))
    wall_limit = sim.track.length / 2 - sim.vehicle.length / 2
    assert tr.poses[:, 0].max() == pytest.approx(wall_limit, abs=.01)
    assert abs(tr.velocities[-1, 0]) < .02
    assert tr.poses[:, 2].min() > sim.vehicle.height / 2


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA driver unavailable")
def test_cuda_driving_accelerates_and_brakes():
    sim = Simulation(scenario="accelerate-brake", device="cuda:0")
    tr = sim.run(5)
    assert np.isfinite(tr.poses).all()
    assert tr.velocities[:, 0].max() > 1
    assert abs(tr.velocities[-1, 0]) < .05
