"""Behavior checks for gravity and contacts; no sensor test suite is run here."""
import itertools

import numpy as np
import pytest
import warp as wp

from warptracer.scene import Track, Vehicle
from warptracer.simulation import Simulation


def world_corners(poses, dimensions):
    corners = np.asarray(list(itertools.product((-1, 1), repeat=3))) * np.asarray(dimensions) / 2
    xyz = poses[:, None, :3]
    q = poses[:, None, 3:6]
    w = poses[:, None, 6:7]
    # Quaternion-vector rotation, independent of Newton's contact calculations.
    t = 2 * np.cross(q, corners)
    return xyz + corners + w * t + np.cross(q, t)


def test_free_fall_before_contact_and_reset():
    sim = Simulation(device="cpu")
    q0, _ = sim.snapshot()
    trajectory = sim.run(duration=.1, record_fps=30)
    t = trajectory.times[-1]
    expected_z = q0[2] - .5 * 9.81 * t * t
    assert trajectory.poses[-1, 2] == pytest.approx(expected_z, abs=.003)
    assert trajectory.velocities[-1, 2] == pytest.approx(-9.81 * t, abs=.002)
    sim.reset()
    np.testing.assert_allclose(sim.snapshot()[0], q0, atol=1e-6)
    assert float(sim.model.body_mass.numpy()[sim.body]) == pytest.approx(sim.vehicle.mass)


def test_tilted_chassis_settles_without_falling_through_floor():
    sim = Simulation(device="cpu")
    trajectory = sim.run(duration=3)
    corners = world_corners(trajectory.poses, sim.vehicle.dimensions)
    assert corners[..., 2].min() > -.015
    # Iterative contact resolution allows sub-millimeter resting penetration.
    assert abs(corners[-1, :, 2].min()) < .002
    assert np.linalg.norm(trajectory.velocities[-1, :3]) < .05
    assert np.linalg.norm(trajectory.velocities[-1, 3:]) < .1
    np.testing.assert_allclose(np.linalg.norm(trajectory.poses[:, 3:], axis=1), 1, atol=1e-5)


def test_outer_barrier_stops_sideways_impact():
    sim = Simulation(device="cpu", scenario="wall-impact")
    trajectory = sim.run(duration=2)
    corners = world_corners(trajectory.poses, sim.vehicle.dimensions)
    wall_face = -(sim.track.bend_radius + sim.track.lane_width / 2) + sim.track.barrier_thickness / 2
    # Require an actual approach to the wall, then no passage through it.
    assert corners[..., 1].min() < wall_face + .05
    assert corners[..., 1].min() >= wall_face - .015
    assert abs(trajectory.velocities[-1, 1]) < .1


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        Track(bend_radius=.5, lane_width=2)
    with pytest.raises(ValueError):
        Vehicle(mass=0)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA driver unavailable")
def test_cuda_drop_remains_finite_and_above_floor():
    sim = Simulation(device="cuda:0")
    trajectory = sim.run(duration=2)
    assert np.isfinite(trajectory.poses).all()
    assert world_corners(trajectory.poses, sim.vehicle.dimensions)[..., 2].min() > -.025
