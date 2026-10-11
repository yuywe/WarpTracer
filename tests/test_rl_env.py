"""Batched episodes, selective scans, wrapped progress and Torch stream ownership."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
import warp as wp
from dataclasses import replace
from warptracer.rl_env import EpisodeConfig, RacingEnv
from warptracer.distance_field import GridScene, DistanceField
from warptracer.lidar import LidarConfig
from warptracer.terrain import OvalTrack

DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not wp.is_cuda_available() or not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
def test_selective_scan_preserves_survivor_ranges_and_statistics(device):
    scene = GridScene(DistanceField.oval(OvalTrack()), device)
    sensor = scene.sensor(LidarConfig().rays(), batch_size=3)
    poses = np.array([OvalTrack().pose(a, .35) for a in (.2, .5, .9)])
    sensor.scan(poses[:, :3])
    old = sensor.values.numpy().copy()
    counts = sensor.iteration_maxima.numpy().copy()
    new_positions = poses[:, :3].copy(); new_positions[:, 0] += .1
    sensor.origins.assign(new_positions)
    sensor.scan_device(sensor.origins, sensor.rotations, wp.array([0, 1, 0], dtype=int, device=device))
    new = sensor.values.numpy()
    np.testing.assert_array_equal(old[[0, 2]], new[[0, 2]])
    np.testing.assert_array_equal(counts[[0, 2]], sensor.iteration_maxima.numpy()[[0, 2]])
    assert not np.array_equal(old[1], new[1])


@pytest.mark.parametrize('device', DEVICES)
def test_timeout_has_fresh_final_observation_and_only_resets_finished_car(device):
    env = RacingEnv(EpisodeConfig(num_envs=3, max_steps=4), device=device)
    env.tensors['lengths'][1] = 3
    before = env.tensors['episodes'].clone()
    env.step(torch.ones(3, 2, device=env.torch_device) * .3)
    assert env.tensors['terminated'].tolist() == [0, 0, 0]
    assert env.tensors['truncated'].tolist() == [0, 1, 0]
    assert env.tensors['episodes'].tolist() == (before + torch.tensor([0, 1, 0], device=env.torch_device)).tolist()
    assert env.tensors['lengths'].tolist() == [1, 0, 1]
    assert not torch.equal(env.observation[1], env.final_observation[1])
    torch.testing.assert_close(env.observation[[0, 2]], env.final_observation[[0, 2]])
    assert env.observation[1, 108].item() == pytest.approx(env.config.start_speed/env.config.max_speed)


@pytest.mark.parametrize('device', DEVICES)
def test_collision_is_terminal_and_penalized_even_if_timeout_coincides(device):
    env = RacingEnv(EpisodeConfig(num_envs=1, max_steps=1), device=device)
    env.sim.state.body_q.assign(np.array([[7.98, 0, .35, 0, 0, 0, 1]], np.float32))
    env.sim.lean_motion.assign(np.array([[1., 0., 0., 0.]], np.float32))
    _, reward, term, trunc = env.step(torch.tensor([[0., 1.]], device=env.torch_device))
    assert term[0] == 1 and trunc[0] == 0 and env.tensors['reason'][0] == 1
    assert reward[0] == -env.config.collision_penalty
    assert env.tensors['finished_length'][0] == 1 and env.tensors['lengths'][0] == 0
    assert torch.isfinite(env.observation).all()


def test_progress_wraps_signed_and_lap_completion_is_terminal():
    env = RacingEnv(EpisodeConfig(num_envs=1, start_speed=0), device='cpu')
    pose = env.track.pose(-np.pi+.01, env.sim.drive.ride_height(env.sim.vehicle))
    env.sim.state.body_q.assign(pose[None])
    env.sim.lean_motion.assign(np.array([[0,0,0, np.arctan2(-4., 0.)]], np.float32))
    env.tensors['previous'][0] = np.pi-.01
    env.step(torch.zeros(1, 2))
    assert env.tensors['progress'][0].item() == pytest.approx(.02, abs=1e-5)
    assert env.tensors['reward'][0] > 0
    env.tensors['progress'][0] = 2*np.pi+.01
    env.step(torch.zeros(1, 2))
    assert env.tensors['reason'][0] == 2 and env.tensors['terminated'][0] == 1


@pytest.mark.parametrize('device', DEVICES)
def test_eager_graph_episode_and_seed_parity(device):
    config = EpisodeConfig(num_envs=8, max_steps=7, seed=9)
    a, b = [RacingEnv(config, device=device, backend=backend) for backend in ('eager', 'graph')]
    rng = np.random.default_rng(3)
    for _ in range(20):
        actions = torch.tensor(rng.uniform(-1,1,(8,2)),dtype=torch.float32,device=a.torch_device)
        x = tuple(v.clone() for v in a.step(actions)); y = b.step(actions)
        for xv, yv in zip(x,y): torch.testing.assert_close(xv,yv,atol=2e-5,rtol=2e-5)
    assert (a.tensors['episodes'] == 3).all()  # initial + two timeouts
    for env in (a,b):
        env.reset()
    torch.testing.assert_close(a.observation,b.observation)


@pytest.mark.skipif(not wp.is_cuda_available() or not torch.cuda.is_available(), reason='CUDA unavailable')
def test_nondefault_torch_stream_orders_actions_physics_and_observations():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        env = RacingEnv(EpisodeConfig(num_envs=4), device='cuda:0')
        actions = torch.zeros(4,2,device='cuda:0'); actions[:,1] = 1
        env.step(actions)
        copied = env.observation.clone()
    torch.cuda.current_stream().wait_stream(stream)
    assert (copied[:,108] > env.config.start_speed/env.config.max_speed).all()


@pytest.mark.parametrize('device', DEVICES)
def test_4096_car_episode_masks_and_buffer_shapes(device):
    env = RacingEnv(EpisodeConfig(num_envs=4096, max_steps=4), device=device)
    indices = torch.tensor([0,257,1023,4095], device=env.torch_device)
    env.tensors['lengths'][indices] = 3
    env.step(torch.zeros(4096,2,device=env.torch_device))
    assert env.observation.shape == (4096,114)
    assert env.tensors['truncated'].sum().item() == 4
    assert (env.tensors['episodes'][indices] == 2).all()
    assert env.tensors['terminated'].sum().item() == 0
    mask = torch.ones(4096,dtype=torch.bool,device=env.torch_device); mask[indices] = False
    torch.testing.assert_close(env.observation[mask],env.final_observation[mask])
    assert torch.isfinite(env.observation).all()


def test_nonfinite_motion_terminates_and_resets_without_nan_observations():
    env = RacingEnv(EpisodeConfig(num_envs=1),device='cpu')
    env.sim.lean_motion.assign(np.array([[float('nan'),0,0,0]],np.float32))
    _,reward,term,trunc = env.step(torch.zeros(1,2))
    assert term[0] == 1 and trunc[0] == 0 and env.tensors['reason'][0] == 5
    assert reward[0] == -5
    assert torch.isfinite(env.observation).all() and torch.isfinite(env.final_observation).all()
    assert torch.isfinite(env.tensors['finished_laps']).all()
    assert np.isfinite(env.sim.state.body_q.numpy()).all()
