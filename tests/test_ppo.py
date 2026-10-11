"""Independent GAE, transformed action likelihood, clipping, update and resume checks."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip('torch')
import warp as wp
from warptracer.ppo import ActorCritic, PPO, PPOConfig, generalized_advantages, clipped_loss
from warptracer.rl_env import RacingEnv, EpisodeConfig
from warptracer.train import save_checkpoint, main

torch.set_num_threads(2)


def test_gae_bootstraps_timeouts_but_never_crosses_an_episode_boundary():
    rewards = torch.tensor([[1., 1.], [100., 100.]])
    values = torch.tensor([[2., 2.], [3., 3.]])
    next_values = torch.tensor([[5., 5.], [7., 7.]])
    terminated = torch.tensor([[1., 0.], [0., 0.]])
    truncated = torch.tensor([[0., 1.], [0., 0.]])
    adv, ret = generalized_advantages(rewards, values, next_values, terminated, truncated, .9, .8)
    # Failure: 1 - 2. Timeout: 1 + .9*5 - 2. Neither takes the next episode's 103.3 delta.
    torch.testing.assert_close(adv, torch.tensor([[-1., 3.5], [103.3, 103.3]]))
    torch.testing.assert_close(ret, adv + values)
    # Ordinary rollout boundary still bootstraps and carries the GAE trace.
    adv, _ = generalized_advantages(rewards[:, :1], values[:, :1], next_values[:, :1],
                                   torch.zeros(2, 1), torch.zeros(2, 1), .9, .8)
    assert adv[0, 0] == pytest.approx(3.5 + .9*.8*103.3)


def test_squashed_likelihood_matches_independent_torch_transform_and_saturation_is_finite():
    torch.manual_seed(6)
    agent = ActorCritic(obs_dim=4, hidden=16)
    obs = torch.randn(7, 4)
    latent = torch.randn(7, 2)
    actual, _, _ = agent.evaluate(obs, latent)
    transformed = torch.distributions.TransformedDistribution(agent.distribution(obs),
        [torch.distributions.transforms.TanhTransform(cache_size=1)])
    expected = transformed.log_prob(latent.tanh()).sum(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    action, z, logp, value = agent.sample(obs)
    assert (action.abs() <= 1).all()
    torch.testing.assert_close(agent.evaluate(obs, z)[0], logp)
    saturated = torch.full((7, 2), 30.)
    logp, _, _ = agent.evaluate(obs, saturated)
    assert torch.isfinite(logp).all()
    logp.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in agent.actor.parameters())


def test_surrogate_clips_positive_and_negative_advantages():
    class Fixed:
        def evaluate(self, obs, latent):
            return torch.log(torch.tensor([1.5, .5])), torch.zeros(2), torch.zeros(2)
    loss, stats = clipped_loss(Fixed(), None, None, torch.zeros(2), torch.zeros(2),
                               torch.tensor([1., -1.]), torch.zeros(2), PPOConfig())
    # min(1.5,1.2) and min(-.5,-.8) => -(1.2-.8)/2.
    assert loss.item() == pytest.approx(-.2)
    assert stats[4] == 1


def small_trainer():
    env = RacingEnv(EpisodeConfig(num_envs=4, max_steps=11), device='cpu')
    return PPO(env, PPOConfig(rollout_steps=8, hidden=32, epochs=2, minibatches=2))


def test_ppo_rejects_recording_mode_to_prevent_stepping_finished_episodes():
    env = RacingEnv(EpisodeConfig(num_envs=1), device='cpu', auto_reset=False)
    with pytest.raises(ValueError, match='auto-reset'):
        PPO(env)


def test_policy_updates_are_finite_and_rollout_observations_are_copies():
    torch.manual_seed(2)
    trainer = small_trainer()
    initial = {k: v.clone() for k, v in trainer.agent.state_dict().items()}
    result = trainer.iterate()
    assert all(np.isfinite(v) for v in result.values() if isinstance(v, (float, int)))
    assert result['transitions'] == 32 and result['lidar_exhaustions'] == 0
    assert any(not torch.equal(v, trainer.agent.state_dict()[k]) for k, v in initial.items())
    stored = trainer.buffer['observations'].clone()
    trainer.env.step(torch.ones(4, 2))
    torch.testing.assert_close(trainer.buffer['observations'], stored)


def test_checkpoint_resumes_optimizer_rng_and_each_episode_exactly(tmp_path):
    torch.manual_seed(4)
    trainer = small_trainer()
    trainer.iterate()
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(trainer, path)
    state = torch.load(path, weights_only=True)
    # Checkpoints made before the recording-mode option remain resumable.
    state['environment'].pop('auto_reset', None)
    expected = trainer.iterate()
    parameters = copy.deepcopy(trainer.agent.state_dict())
    positions = trainer.env.sim.state.body_q.numpy()
    resumed = small_trainer()
    resumed.load_state_dict(state)
    actual = resumed.iterate()
    for k, v in parameters.items():
        torch.testing.assert_close(v, resumed.agent.state_dict()[k], rtol=0, atol=0)
    np.testing.assert_array_equal(positions, resumed.env.sim.state.body_q.numpy())
    for key in ('transitions', 'mean_step_reward', 'approx_kl', 'mean_episode_return'):
        assert actual[key] == expected[key]


def test_smoke_cli_saves_loadable_checkpoint_and_jsonl(tmp_path):
    main(['smoke', '--updates', '1', '--output', str(tmp_path)])
    state = torch.load(tmp_path / 'checkpoint.pt', weights_only=True)
    assert state['iteration'] == 1 and state['global_step'] == 256
    main(['smoke', '--updates', '1', '--resume', str(tmp_path / 'checkpoint.pt'), '--output', str(tmp_path)])
    assert torch.load(tmp_path / 'checkpoint.pt', weights_only=True)['iteration'] == 2
    assert len((tmp_path / 'training.jsonl').read_text().splitlines()) == 2


@pytest.mark.skipif(not wp.is_cuda_available() or not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_rollout_and_optimizer_on_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        env = RacingEnv(EpisodeConfig(num_envs=8,max_steps=13), device='cuda:0')
        trainer = PPO(env,PPOConfig(rollout_steps=8,epochs=1,minibatches=2,hidden=32))
        metrics = trainer.iterate()
    assert metrics['transitions'] == 64 and metrics['lidar_exhaustions'] == 0
    assert np.isfinite(metrics['approx_kl'])
