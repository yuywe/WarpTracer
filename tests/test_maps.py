"""PNG coordinates, route geometry, shared sensing/collision and portable PPO."""
import json

import numpy as np
from PIL import Image
import pytest
import warp as wp

from warptracer.maps import load_map, restore_track, track_state, main as inspect_map
from warptracer.scene import Vehicle

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA unavailable"))]


@pytest.fixture
def png_map(tmp_path):
    # A rectangular ring, deliberately offset from the origin and not elliptical.
    free = np.zeros((240, 160), bool)
    free[10:230, 10:150] = True
    free[50:190, 50:110] = False
    image = tmp_path / "circuit.png"
    Image.fromarray(np.flipud(free.T).astype(np.uint8) * 255).save(image)
    config = tmp_path / "circuit.json"
    config.write_text(json.dumps({"image": image.name, "resolution": .05, "origin": [-3., 4.]}))
    return config, free


def test_png_import_coordinates_scale_route_and_checkpoint(png_map, tmp_path):
    config, free = png_map
    track = load_map(config.with_suffix(".png"))
    np.testing.assert_array_equal(track.free, free)
    assert (track.length, track.width) == (12., 8.) and track.origin == (-3., 4.)
    assert 25 < track.route_length < 35
    assert track.validate_vehicle(Vehicle()) > .6
    area = np.sum(track.route[:, 0] * np.roll(track.route[:, 1], -1) -
                  track.route[:, 1] * np.roll(track.route[:, 0], -1))
    assert area > 0
    state = track_state(track)
    config.unlink(); config.with_suffix(".png").unlink()
    rebuilt = restore_track(state)
    assert track_state(rebuilt) == state
    np.testing.assert_array_equal(track.clearance, rebuilt.clearance)
    field = rebuilt.distance_field()
    np.testing.assert_array_equal(field.walls[2:-2, 2:-2], ~free)
    assert field.origin == pytest.approx((-3.075, 3.925))


def test_preview_and_reverse_route(png_map, tmp_path):
    config, _ = png_map
    track = load_map(config)
    settings = json.loads(config.read_text()); settings["reverse"] = True
    config.write_text(json.dumps(settings))
    reverse = load_map(config)
    np.testing.assert_array_equal(reverse.route, track.route[::-1])
    output = tmp_path / "preview.png"
    inspect_map([str(config), "--output", str(output)])
    assert Image.open(output).size == (240, 160)


def test_unknown_transparent_and_gray_pixels_are_obstacles(png_map):
    config, _ = png_map
    route = load_map(config).route
    np.savetxt(config.parent / "route.csv", route, delimiter=",")
    settings = json.loads(config.read_text()); settings["centerline"] = "route.csv"
    config.write_text(json.dumps(settings))
    image = Image.open(config.with_suffix(".png")).convert("RGBA")
    # Pixel XY=(35,30) is in the lane, clear of the extracted centerline.
    image.putpixel((35, 129), (255, 255, 255, 0))
    image.putpixel((36, 129), (127, 127, 127, 255))
    image.putpixel((37, 129), (205, 205, 205, 255))
    image.save(config.with_suffix(".png"))
    track = load_map(config)
    assert not track.free[35:38, 30].any()


def test_invalid_topology_and_scale_fail_clearly(png_map):
    config, free = png_map
    settings = json.loads(config.read_text()); settings["resolution"] = 0
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="resolution"):
        load_map(config)
    settings["resolution"] = .001
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="clearance|two meters"):
        load_map(config).validate_vehicle(Vehicle())
    settings["resolution"] = .05
    config.write_text(json.dumps(settings))
    free[50:190, 50:110] = True  # room without a closed racing route
    Image.fromarray(np.flipud(free.T).astype(np.uint8)*255).save(config.with_suffix(".png"))
    with pytest.raises(ValueError, match="closed route"):
        load_map(config)


def test_route_extraction_tolerates_a_small_boundary_defect(png_map):
    # Real track PNGs rarely skeletonize to a perfect ring: a small island
    # obstacle fully inside the lane (e.g. a rough edge at the start/finish
    # line) locally forks the skeleton into two short parallel paths. This
    # must still resolve to one route rather than rejecting the whole map.
    config, free = png_map
    free[22:38, 68:76] = False
    Image.fromarray(np.flipud(free.T).astype(np.uint8) * 255).save(config.with_suffix(".png"))
    track = load_map(config)
    assert 25 < track.route_length < 35


def test_explicit_csv_route_for_room_and_occupied_border(png_map):
    config, free = png_map
    free[50:190, 50:110] = True
    Image.fromarray(np.flipud(free.T).astype(np.uint8)*255).save(config.with_suffix(".png"))
    np.savetxt(config.parent / "route.csv", [[-1, 6], [7, 6], [7, 10], [-1, 10]], delimiter=",")
    settings = json.loads(config.read_text()); settings["centerline"] = "route.csv"
    config.write_text(json.dumps(settings))
    track = load_map(config)
    assert track.validate_vehicle(Vehicle()) > .8
    free[0, 70] = True
    Image.fromarray(np.flipud(free.T).astype(np.uint8)*255).save(config.with_suffix(".png"))
    with pytest.raises(ValueError, match="border"):
        load_map(config)


def test_conservative_clearance_covers_the_entire_occupied_pixel(png_map):
    track = load_map(png_map[0])
    # First free column starts at -2.5 m; exact vertical wall distance here is .5.
    p = np.array([[-2., 7.]])
    assert 0 < track.clearance_at(p)[0] <= .5
    assert track.clearance_at(np.array([[-4., 7.]]))[0] < 0


@pytest.mark.parametrize("device", DEVICES)
def test_png_lidar_hits_image_wall_in_world_coordinates(png_map, device):
    import warp as wp
    from warptracer.lidar import LidarConfig, MountedLidar
    from warptracer.lean import LeanState
    track = load_map(png_map[0])
    scanner = MountedLidar(track, Vehicle(), LidarConfig(backend="grid", beams=1,
                           mount_position=(0, 0, .2)), device)
    state = LeanState(device)
    state.body_q.assign(np.array([[-1., 5.5, .1, 0, 0, 0, 1]], np.float32))
    scanner.update(state, 0, 0)
    _, ranges, valid = scanner.snapshot()
    assert valid[0] and ranges[0] == pytest.approx(9.5, abs=.015)
    assert scanner.diagnostics()["iteration_limit_events"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_map_footprint_collision_and_fused_substep_parity(png_map, device):
    from warptracer.simulation import Simulation
    track = load_map(png_map[0])
    a, b = [Simulation(track=track, scenario="drive", engine="lean", device=device) for _ in range(2)]
    for sim in (a, b):
        sim.set_action(throttle=1, steering=.1)
    a._fused_physics_steps(4)
    for _ in range(4): b._physics_step()
    np.testing.assert_allclose(a.state.body_q.numpy(), b.state.body_q.numpy(), atol=1e-7)
    # Center remains in a white pixel, but the full car's collision circle hits.
    a.state.body_q.assign(np.array([[-2.3, 7, .1, 0, 0, 0, 1]], np.float32))
    a.lean_motion.assign(np.array([[1., 0, 0, 0]], np.float32))
    a._fused_physics_steps(4)
    assert a.wall_contact_substeps.numpy()[0] > 0
    assert a.lean_motion.numpy()[0, 0] == 0


def test_swept_collision_catches_thin_wall_between_safe_endpoints(png_map):
    from warptracer.simulation import Simulation
    config, free = png_map
    free[120, 20:40] = False  # one-pixel wall across part of the lower straight
    Image.fromarray(np.flipud(free.T).astype(np.uint8)*255).save(config.with_suffix(".png"))
    np.savetxt(config.parent / "route.csv", [[-2, 6], [-1, 6], [-1, 9], [-2, 9]], delimiter=",")
    settings = json.loads(config.read_text()); settings["centerline"] = "route.csv"
    config.write_text(json.dumps(settings))
    sim = Simulation(track=load_map(config), scenario="drive", engine="lean", device="cpu")
    initial = np.array([[2., 5.5, .1, 0, 0, 0, 1]], np.float32)
    sim.state.body_q.assign(initial)
    # Deliberately beyond supported racing speeds: a one-substep jump across
    # the wall would end in free space if only its endpoint were checked.
    sim.lean_motion.assign(np.array([[500., 0, 0, 0]], np.float32))
    sim._physics_step()
    assert sim.wall_contact_substeps.numpy()[0] == 1
    np.testing.assert_array_equal(sim.state.body_q.numpy(), initial)


def test_png_map_setup_creates_sidecar_once_and_supports_inversion(png_map, tmp_path):
    config, _ = png_map
    original = config.read_text()
    with pytest.raises(SystemExit):
        inspect_map([str(config.with_suffix(".png")), "--resolution", ".1"])
    assert config.read_text() == original
    config.unlink()
    inspect_map([str(config.with_suffix(".png")), "--resolution", ".05",
                 "--output", str(tmp_path / "preview.png")])
    a = load_map(config)
    image = np.asarray(Image.open(config.with_suffix(".png")))
    Image.fromarray(255-image).save(config.with_suffix(".png"))
    settings = json.loads(config.read_text()); settings["negate"] = True
    config.write_text(json.dumps(settings))
    b = load_map(config)
    np.testing.assert_array_equal(a.free, b.free)
    np.testing.assert_array_equal(a.route, b.route)


@pytest.mark.parametrize("device", DEVICES)
def test_map_graph_parity_resets_signed_progress_and_lap(png_map, device):
    torch = pytest.importorskip("torch")
    from warptracer.rl_env import RacingEnv, EpisodeConfig
    track = load_map(png_map[0])
    a, b = [RacingEnv(EpisodeConfig(num_envs=4, max_steps=5), track=track,
                       backend=backend, device=device) for backend in ("eager", "graph")]
    actions = torch.tensor([[0., .5]]*4, device=a.torch_device)
    for _ in range(12):
        x = [v.clone() for v in a.step(actions)]
        y = b.step(actions)
        for left, right in zip(x, y): torch.testing.assert_close(left, right)
    assert torch.isfinite(a.observation).all()
    assert (a.tensors["episodes"] == 3).all()
    # Put a stationary car just past the closing segment, with the previous
    # position just before it. Reward is continuous across the route wrap.
    pose = track.path(a.sim.drive.ride_height(a.sim.vehicle))[0]
    a.sim.state.body_q.assign(np.tile(pose, (4, 1)))
    a.sim.lean_motion.zero_()
    a.route_index.fill_(0)
    a.tensors["previous"].fill_(2*np.pi-.01)
    a.tensors["progress"].zero_()
    a.step(torch.zeros(4, 2, device=a.torch_device))
    assert (a.tensors["reward"] > 0).all()
    assert a.tensors["progress"][0].item() == pytest.approx(.01, abs=1e-5)
    a.tensors["progress"].fill_(2*np.pi+.01)
    a.step(torch.zeros(4, 2, device=a.torch_device))
    assert (a.tensors["reason"] == 2).all()


def test_map_resume_preserves_route_cursor_and_selective_reset(png_map):
    torch = pytest.importorskip("torch")
    from warptracer.rl_env import RacingEnv, EpisodeConfig
    track = load_map(png_map[0])
    config = EpisodeConfig(num_envs=3, max_steps=100)
    env = RacingEnv(config, track=track, device="cpu")
    actions = torch.tensor([[0., .5]]*3)
    for _ in range(10): env.step(actions)
    state = env.state_dict()
    rebuilt = RacingEnv(config, track=restore_track(state["track"]), device="cpu")
    rebuilt.load_state_dict(state)
    old = env.tensors["episodes"].clone()
    for candidate in (env, rebuilt):
        candidate.tensors["lengths"][1] = 99
    actual = [x.clone() for x in env.step(actions)]
    expected = rebuilt.step(actions)
    for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
    np.testing.assert_array_equal(env.route_index.numpy(), rebuilt.route_index.numpy())
    assert env.tensors["episodes"].tolist() == (old + torch.tensor([0, 1, 0])).tolist()
    torch.testing.assert_close(env.observation[[0, 2]], env.final_observation[[0, 2]])
    # Signed backward movement across the wrap produces a negative reward.
    p = track.path(env.sim.drive.ride_height(env.sim.vehicle))[0]
    env.sim.state.body_q.assign(np.tile(p, (3, 1)))
    env.sim.lean_motion.zero_(); env.route_index.zero_()
    env.tensors["previous"].fill_(.01)
    env.step(torch.zeros(3, 2))
    assert (env.tensors["reward"] < 0).all()


def test_map_ppo_checkpoint_resume_and_evaluate_without_source_png(png_map, tmp_path):
    torch = pytest.importorskip("torch")
    from warptracer.train import main, evaluate
    config, _ = png_map
    output = tmp_path / "ppo"
    main(["smoke", "--updates", "1", "--map", str(config), "--output", str(output)])
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    assert saved["environment"]["track"]["kind"] == "png"
    config.unlink(); config.with_suffix(".png").unlink()
    main(["smoke", "--updates", "1", "--resume", str(output / "checkpoint.pt"), "--output", str(output)])
    state = torch.load(output / "checkpoint.pt", weights_only=True)
    assert state["iteration"] == 2
    # Short recording verifies map restoration and rendering without a PNG path.
    state["environment"]["config"]["max_steps"] = 1
    torch.save(state, output / "checkpoint.pt")
    evaluate([str(output / "checkpoint.pt"), "--device", "cpu", "--output", str(output / "evaluation")])
    metrics = json.loads((output / "evaluation.json").read_text())
    assert metrics["map"] == "circuit" and metrics["transitions"] == 1
    from warptracer.playback import load_recording
    recording, track, vehicle = load_recording(output / "evaluation.npz")
    assert track.name == "circuit" and recording.lidar.directions.shape == (108, 3)
