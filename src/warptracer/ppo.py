"""Clipped PPO with bounded continuous actions and timeout-aware GAE.

Independent implementation informed by the PPO/GAE papers and the two racing
references documented in docs/ppo.md. Environment work stays in Warp; policy
inference, rollout storage, gradients and Adam stay on the same Torch device.
"""
from dataclasses import dataclass, asdict
import math
import time

import torch
from torch import nn
from torch.nn import functional as F
import warp as wp


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 64
    epochs: int = 4
    minibatches: int = 8
    hidden: int = 128
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    value_clip: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = 0.0  # optional base-Gaussian entropy surrogate
    max_grad_norm: float = .5
    target_kl: float = .02

    def __post_init__(self):
        for name in ("rollout_steps", "epochs", "minibatches", "hidden"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        floats = [v for v in asdict(self).values() if isinstance(v, float)]
        if not all(math.isfinite(v) for v in floats):
            raise ValueError("PPO parameters must be finite")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Require 0 < gamma <= 1 and 0 <= gae_lambda <= 1")
        if not 0 < self.clip < 1 or min(self.learning_rate, self.max_grad_norm, self.target_kl) <= 0:
            raise ValueError("Require positive optimizer limits and 0 < clip < 1")
        if min(self.value_clip, self.value_coefficient, self.entropy_coefficient) < 0:
            raise ValueError("Loss coefficients must be nonnegative")


def _network(inputs, hidden, outputs, head_gain):
    layers = [nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs)]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, gain=head_gain)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=114, action_dim=2, hidden=128):
        super().__init__()
        self.actor = _network(obs_dim, hidden, action_dim, .01)
        self.critic = _network(obs_dim, hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), -.5))

    def value(self, observations):
        return self.critic(observations).squeeze(-1)

    def distribution(self, observations):
        return torch.distributions.Normal(self.actor(observations), self.log_std.clamp(-4, .5).exp(),
                                          validate_args=False)

    @staticmethod
    def _log_probability(dist, latent):
        # Stable log(1 - tanh(z)^2), even for saturated actions. Storing latent
        # samples avoids inverse-tanh rounding; the same transform is used in PPO.
        jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return (dist.log_prob(latent) - jacobian).sum(-1)

    def evaluate(self, observations, latent):
        dist = self.distribution(observations)
        return self._log_probability(dist, latent), dist.entropy().sum(-1), self.value(observations)

    def sample(self, observations, deterministic=False):
        dist = self.distribution(observations)
        latent = dist.mean if deterministic else dist.sample()
        return latent.tanh(), latent, self._log_probability(dist, latent), self.value(observations)


@torch.jit.script
def generalized_advantages(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
                           terminated: torch.Tensor, truncated: torch.Tensor,
                           gamma: float, lam: float):
    """Bootstrap time limits from their pre-reset value, cut traces at ALL ends."""
    bootstrap = torch.where((terminated != 0), torch.zeros_like(next_values), next_values)
    deltas = rewards + gamma * bootstrap - values
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        continuation = ((terminated[t] == 0) & (truncated[t] == 0)).to(dtype=rewards.dtype)
        carry = deltas[t] + gamma * lam * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values


def clipped_loss(agent, obs, latent, old_logp, old_value, advantage, returns, config):
    new_logp, entropy, value = agent.evaluate(obs, latent)
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1-config.clip, 1+config.clip) * advantage).mean()
    error = (value - returns).square()
    if config.value_clip > 0:
        clipped = old_value + (value - old_value).clamp(-config.value_clip, config.value_clip)
        error = torch.maximum(error, (clipped - returns).square())
    value_loss = .5 * error.mean()
    loss = policy + config.value_coefficient * value_loss - config.entropy_coefficient * entropy.mean()
    metrics = torch.stack([policy, value_loss, entropy.mean(),
                           ((ratio-1) - log_ratio).mean(), ((ratio-1).abs() > config.clip).float().mean()])
    return loss, metrics.detach()


class PPO:
    def __init__(self, env, config=None):
        self.env, self.config = env, config or PPOConfig()
        self.agent = ActorCritic(env.obs_dim, env.action_dim, self.config.hidden).to(env.torch_device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.iteration, self.global_step = 0, 0
        t, n = self.config.rollout_steps, env.num_envs
        self.batch_size = t * n
        if self.batch_size % self.config.minibatches or self.batch_size < 2:
            raise ValueError("Rollout batch must be >= 2 and divisible by minibatches")
        self.buffer = {"observations": torch.empty((t, n, env.obs_dim), device=env.torch_device),
                       "latent": torch.empty((t, n, env.action_dim), device=env.torch_device)}
        for key in ("logp", "value", "next_value", "reward", "terminated", "truncated"):
            self.buffer[key] = torch.empty((t, n), device=env.torch_device)
        self.episode_stats = torch.zeros(5, device=env.torch_device)

    def _sync(self):
        if self.env.device.is_cuda:
            torch.cuda.synchronize(self.env.torch_device)

    @torch.no_grad()
    def collect(self):
        self.episode_stats.zero_()
        with self.env._scope():
            self.env.sim.lidar.reset_statistics()
        b, e = self.buffer, self.env
        for t in range(self.config.rollout_steps):
            # Public env outputs are borrowed buffers, so store a COPY before step.
            b["observations"][t].copy_(e.observation)
            action, latent, logp, value = self.agent.sample(b["observations"][t])
            b["latent"][t].copy_(latent)
            b["logp"][t].copy_(logp)
            b["value"][t].copy_(value)
            _, reward, terminated, truncated = e.step(action)
            b["reward"][t].copy_(reward)
            b["terminated"][t].copy_(terminated)
            b["truncated"][t].copy_(truncated)
            # Fresh next-state scan BEFORE any auto-reset. Terminal values are
            # masked by GAE; time-limit values remain legitimate bootstraps.
            b["next_value"][t].copy_(self.agent.value(e.final_observation))
            self.episode_stats += torch.stack([e.tensors["done"].sum(), e.tensors["finished_return"].sum(),
                e.tensors["finished_length"].sum(), (e.tensors["reason"] == 2).sum(),
                e.tensors["finished_laps"].sum()])
        return generalized_advantages(b["reward"], b["value"], b["next_value"], b["terminated"],
                                       b["truncated"], self.config.gamma, self.config.gae_lambda)

    def update(self, advantages, returns):
        c, b = self.config, self.buffer
        size = self.batch_size // c.minibatches
        obs = b["observations"].flatten(0, 1)
        latent = b["latent"].flatten(0, 1)
        adv = advantages.flatten()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        ret, value, logp = returns.flatten(), b["value"].flatten(), b["logp"].flatten()
        accumulated = torch.zeros(5, device=obs.device)
        updates, stopped = 0, False
        for _ in range(c.epochs):
            order = torch.randperm(self.batch_size, device=obs.device)
            epoch_kl = torch.zeros((), device=obs.device)
            for start in range(0, self.batch_size, size):
                idx = order[start:start+size]
                loss, metrics = clipped_loss(self.agent, obs[idx], latent[idx], logp[idx], value[idx], adv[idx], ret[idx], c)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), c.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    self.agent.log_std.clamp_(-4, .5)
                accumulated += metrics
                epoch_kl += metrics[3]
                updates += 1
            # One host decision per epoch, never a done.any() within the rollout.
            if epoch_kl.item() / c.minibatches > c.target_kl:
                stopped = True
                break
        return accumulated / updates, stopped

    def iterate(self):
        self._sync()
        start = time.perf_counter()
        advantages, returns = self.collect()
        self._sync()
        collected = time.perf_counter()
        sensor = self.env.sim.lidar.sensor
        exhaustion = int(wp.to_torch(sensor.limit_counts).sum().item())
        if exhaustion:
            raise RuntimeError(f"PPO rollout contained {exhaustion} exhausted LiDAR rays; not updating from incomplete scans")
        if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
            raise RuntimeError("Nonfinite rollout returns/advantages")
        metrics, stopped = self.update(advantages, returns)
        self._sync()
        end = time.perf_counter()
        self.iteration += 1
        self.global_step += self.batch_size
        pg, vf, entropy, kl, clip_fraction = metrics.tolist()
        episodes, ret, length, success, laps = self.episode_stats.tolist()
        return {"iteration": self.iteration, "transitions": self.global_step,
                "policy_loss": pg, "value_loss": vf, "base_gaussian_entropy": entropy,
                "approx_kl": kl, "clip_fraction": clip_fraction, "kl_stop": stopped,
                "mean_step_reward": self.buffer["reward"].mean().item(),
                "completed_episodes": int(episodes), "successful_laps": int(success),
                "mean_episode_return": ret / episodes if episodes else None,
                "mean_episode_length": length / episodes if episodes else None,
                "mean_episode_laps": laps / episodes if episodes else None,
                "rollout_seconds": collected-start, "update_seconds": end-collected,
                "rollout_transitions_per_second": self.batch_size / (collected-start),
                "training_transitions_per_second": self.batch_size / (end-start),
                "lidar_exhaustions": exhaustion,
                "max_march_iterations": int(wp.to_torch(sensor.iteration_maxima).max().item())}

    def state_dict(self):
        return {"format_version": 1, "ppo_config": asdict(self.config), "agent": self.agent.state_dict(),
                "optimizer": self.optimizer.state_dict(), "iteration": self.iteration,
                "global_step": self.global_step, "environment": self.env.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.env.torch_device) if self.env.device.is_cuda else None}

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or state["ppo_config"] != asdict(self.config):
            raise ValueError("Checkpoint PPO settings/version do not match")
        self.agent.load_state_dict(state["agent"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.env.load_state_dict(state["environment"])
        self.iteration, self.global_step = state["iteration"], state["global_step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.env.device.is_cuda and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.env.torch_device)
