"""Two-layer figure eight: stacked surfaces, seams, occlusion, layer switching and parity."""
import json

import numpy as np
import pytest
import warp as wp

from racesense3d import lidar
from racesense3d.core import Scene
from warptracer.benchmark import validate_pair
from warptracer.cli import main
from warptracer.execution import TransitionRunner
from warptracer.layers import SEAM_TOLERANCE, LayeredTrack, figure_eight_bridge
from warptracer.lidar import LidarConfig
from warptracer.maps import restore_track, track_state
from warptracer.scene import Vehicle
from warptracer.simulation import Simulation

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA driver unavailable"))]


@pytest.fixture(scope="module")
def track():
    return figure_eight_bridge()


def car(device="cpu", n=1):
    return Simulation(track=figure_eight_bridge(), scenario="drive", engine="lean", device=device,
                      num_envs=n, lidar=LidarConfig(frequency=60))


def test_bridge_stacks_two_surfaces_over_the_crossing(track):
    assert track.layer_count == 2 and set(np.unique(track.route_layers)) == {0, 1}
    np.testing.assert_allclose(track.height_at(0, [[0., 0.]]), 0, atol=1e-6)
    np.testing.assert_allclose(track.height_at(1, [[0., 0.]]), .7, atol=1e-6)
    # Both layers are drivable at the two seams, where they are one surface,
    # and at the crossing, where the deck is a full bridge height above the road.
    both = track.free[0] & track.free[1]
    gap = np.abs(track.heights[0] - track.heights[1])
    xy = np.array(track.origin) + (np.argwhere(both) + .5) * track.resolution
    seam = gap[both] < SEAM_TOLERANCE
    assert seam.any() and (np.abs(xy[seam, 0]) > 7).all()
    assert (np.linalg.norm(xy[~seam], axis=1) < 2).all() and gap[both][~seam].min() > .6
    assert track.validate_vehicle(Vehicle()) > .5
    grade = np.linalg.norm(track.gradient_at(1, track.route[track.route_layers == 1]), axis=1)
    assert .1 < grade.max() < .15


def test_layered_state_round_trips_through_json(track):
    state = json.loads(json.dumps(track_state(track)))
    assert state["kind"] == "layered" and state["generator"]["bridge_height"] == .7
    np.testing.assert_array_equal(restore_track(state).heights, track.heights)
    # Tracks without a generator (e.g. future scans) embed their grids instead.
    generic = LayeredTrack(track.heights, track.free, track.resolution, track.origin,
                           track.route, track.route_layers)
    copy = restore_track(json.loads(json.dumps(track_state(generic))))
    np.testing.assert_array_equal(copy.clearance, generic.clearance)
    np.testing.assert_array_equal(copy.route_layers, generic.route_layers)


def test_deck_occludes_the_road_below_in_the_lidar_mesh(track):
    scene = Scene(*track.mesh(), device="cpu")
    sensor = scene.sensor(lidar([0.]), near=.01, far=30)
    up = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]]).T     # ray +X -> world +Z
    down = np.array([[0., 0, -1], [0, 1, 0], [1, 0, 0]]).T   # ray +X -> world -Z
    sensor.scan([[0., 0, .19]], up[None])
    assert sensor.result.numpy()[0].item() == pytest.approx(.7 - track.deck_thickness - .19, abs=1e-4)
    sensor.scan([[0., 0, .89]], down[None])
    assert sensor.result.numpy()[0].item() == pytest.approx(.19, abs=1e-4)
    # Looking along the ground straight, the ray passes under the deck.
    dx, dy = track.route[1, :2] - track.route[0, :2]
    c, s = np.array([dx, dy]) / np.hypot(dx, dy)
    sensor.scan([[0., 0, .19]], np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])[None])
    assert sensor.result.numpy()[0].item() > 3


@pytest.mark.parametrize("device", DEVICES)
def test_disparity_drives_under_then_over_the_bridge(device):
    sim = car(device)
    backend = "graph" if device.startswith("cuda") else "eager"
    runner = TransitionRunner(sim, controller="disparity", backend=backend, integrator="fused",
                              substeps=sim.lidar_stride)
    trajectory = sim.run(60, 30, runner=runner)
    q, track = trajectory.poses, sim.track
    assert track.route_progress(q) > 1
    assert sim.wall_contact_substeps.numpy()[0] == 0
    crossing = np.linalg.norm(q[:, :2], axis=1) < 1
    ride = sim.drive.ride_height(sim.vehicle)
    assert (q[crossing, 2] < ride + .05).any() and (q[crossing, 2] > ride + .65).any()
    # Every pose rests on one of the two surfaces, never in between.
    errors = []
    for layer in (0, 1):
        g = track.gradient_at(layer, q)
        errors.append(np.abs(q[:, 2] - track.height_at(layer, q) - ride * np.sqrt(1 + (g * g).sum(1))))
    assert np.min(errors, axis=0).max() < 1e-4
    assert trajectory.lidar.valid.all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend,integrator", [("graph", "unfused"), ("eager", "fused"), ("graph", "fused")])
def test_layered_variants_match_unfused_eager(device, backend, integrator):
    def runner(mode, integration):
        sim = car(device, n=3)
        track = sim.track
        seam = int(np.flatnonzero(np.diff(track.route_layers) == 1)[0])
        # Start of the ground straight, top of the bridge, and 2 m before a seam.
        picks = [0, int(np.argmax(track.route[:, 2])), seam - 40]
        sim.initial_pose[:] = track.path(sim.drive.ride_height(sim.vehicle))[picks]
        sim.initial_layers[:] = track.route_layers[picks]
        return TransitionRunner(sim, controller="disparity", backend=mode, integrator=integration)
    reference, candidate = runner("eager", "unfused"), runner(backend, integrator)
    validate_pair(reference, candidate, transitions=480)
    # validate_pair resets; rerun to check the car that started before the seam.
    assert candidate.sim.layers.numpy().tolist() == [0, 1, 0]
    for _ in range(480):
        candidate.advance()
    assert candidate.sim.layers.numpy()[2] == 1
    assert candidate.sim.wall_contact_substeps.numpy().sum() == 0


def test_short_bridge_command_exports_layered_track(tmp_path):
    main(["demo", "bridge", "--seconds", ".2", "--output", str(tmp_path)])
    report = json.loads((tmp_path / "bridge.json").read_text())
    assert report["track_kind"] == "layered" and report["track"]["kind"] == "layered"
    assert report["completed_laps"] == 0 and report["wall_contact_substeps"] == 0


def test_layered_track_rejects_newton_and_grid_lidar(track):
    with pytest.raises(ValueError, match="lean"):
        Simulation(track=track, scenario="drive", engine="newton")
    with pytest.raises(ValueError, match="Grid LiDAR"):
        Simulation(track=track, scenario="drive", engine="lean", device="cpu",
                   lidar=LidarConfig(backend="grid"))
