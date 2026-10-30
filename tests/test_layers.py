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


def _surface_error(track, poses, layers, ride):
    errors = np.empty(len(poses))
    for layer in (0, 1):
        rows = layers == layer
        g = track.gradient_at(layer, poses[rows])
        errors[rows] = np.abs(poses[rows, 2] - track.height_at(layer, poses[rows])
                              - ride * np.sqrt(1 + (g * g).sum(1)))
    return errors


@pytest.mark.parametrize("device", DEVICES)
def test_ppo_respawns_on_route_layers_and_graph_matches_eager(device):
    torch = pytest.importorskip("torch")
    from warptracer.rl_env import EpisodeConfig, RacingEnv
    track = figure_eight_bridge()
    config = EpisodeConfig(num_envs=32, max_steps=20)
    a, b = [RacingEnv(config, track=track, device=device, backend=mode) for mode in ("eager", "graph")]
    # Each car rests on the surface of the layer its route sample belongs to.
    layers = a.layers.numpy()
    assert set(layers.tolist()) == {0, 1}
    ride = a.sim.drive.ride_height(a.sim.vehicle)
    assert _surface_error(track, a.sim.state.body_q.numpy(), layers, ride).max() < 1e-4
    # Cars spawned on a ramp see its grade in the slope features.
    assert a.observation[:, 112].abs().max().item() > .2
    actions = torch.tensor([[.2 * np.sin(i), .4] for i in range(32)], dtype=torch.float32, device=a.torch_device)
    for _ in range(45):
        x = [v.clone() for v in a.step(actions)]
        y = b.step(actions)
        for left, right in zip(x, y):
            torch.testing.assert_close(left, right)
        assert torch.isfinite(a.observation).all()
        survivors = a.tensors["done"] == 0
        torch.testing.assert_close(a.observation[survivors], a.final_observation[survivors], rtol=0, atol=0)
    assert (a.tensors["episodes"] >= 3).all()
    np.testing.assert_array_equal(a.layers.numpy(), b.layers.numpy())


def test_ppo_route_progress_stays_on_the_bridge_over_the_crossing():
    torch = pytest.importorskip("torch")
    from warptracer.rl_env import EpisodeConfig, RacingEnv
    track = figure_eight_bridge()
    env = RacingEnv(EpisodeConfig(num_envs=1, max_steps=500), track=track, device="cpu", backend="eager")
    # Start 1 m before the crossing on the deck, driving straight over it.
    top = int(np.flatnonzero(track.route_layers == 1)[np.argmin(np.linalg.norm(
        track.route[track.route_layers == 1, :2], axis=1))]) - 20
    pose = track.path(env.sim.drive.ride_height(env.sim.vehicle))[top]
    env.sim.state.body_q.assign(pose[None]); env.sim.next_state.body_q.assign(pose[None])
    q = pose[3:]
    env.sim.lean_motion.assign(np.array([[1., 0., 0., np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]),
                                                                  1 - 2 * (q[1] ** 2 + q[2] ** 2))]], np.float32))
    env.layers.fill_(1)
    env.route_index.fill_(top)
    env.tensors["previous"].fill_(2 * np.pi * float(track.arc[top]) / track.route_length)
    for _ in range(60):
        _, reward, terminated, _ = env.step(torch.tensor([[0., .2]]))
        assert terminated[0] == 0 and reward[0] > 0
    # Two meters on, still on the deck; the route cursor never jumped to the road below.
    index = int(env.route_index.numpy()[0])
    assert track.route_layers[index] == 1 and 20 < index - top < 80
    assert env.layers.numpy()[0] == 1 and env.sim.state.body_q.numpy()[0, 2] > .7


def test_ppo_layered_resume_restores_layers_exactly():
    torch = pytest.importorskip("torch")
    from warptracer.rl_env import EpisodeConfig, RacingEnv
    config = EpisodeConfig(num_envs=6, max_steps=30)
    env = RacingEnv(config, track=figure_eight_bridge(), device="cpu")
    actions = torch.tensor([[.3, .5]] * 6)
    for _ in range(25):
        env.step(actions)
    state = env.state_dict()
    assert state["track"]["kind"] == "layered" and "layers" in state["arrays"]
    rebuilt = RacingEnv(config, track=restore_track(state["track"]), device="cpu")
    rebuilt.load_state_dict(state)
    for _ in range(10):
        expected = [v.clone() for v in env.step(actions)]
        for left, right in zip(expected, rebuilt.step(actions)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    np.testing.assert_array_equal(env.layers.numpy(), rebuilt.layers.numpy())


def test_bridge_ppo_smoke_trains_and_evaluates(tmp_path):
    torch = pytest.importorskip("torch")
    from warptracer.train import evaluate, main as train
    output = tmp_path / "ppo"
    train(["smoke", "--updates", "1", "--bridge", "--output", str(output)])
    state = torch.load(output / "checkpoint.pt", weights_only=True)
    assert state["environment"]["track"]["kind"] == "layered"
    assert json.loads((output / "config.json").read_text())["map"] == "figure-eight-bridge"
    log = json.loads((output / "training.jsonl").read_text().splitlines()[0])
    assert log["lidar_exhaustions"] == 0 and log["max_march_iterations"] is None
    state["environment"]["config"]["max_steps"] = 1
    torch.save(state, output / "checkpoint.pt")
    evaluate([str(output / "checkpoint.pt"), "--device", "cpu", "--output", str(output / "evaluation")])
    metrics = json.loads((output / "evaluation.json").read_text())
    assert metrics["map"] == "figure-eight-bridge" and metrics["march_diagnostics"] is None
    from warptracer.playback import load_recording
    _, track, _ = load_recording(output / "evaluation.npz")
    assert isinstance(track, LayeredTrack)
