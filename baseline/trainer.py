from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from baseline.critic import CriticNetwork
from baseline.policy import IPPOPolicy
from baseline.rollout_buffer import IPPORolloutBuffer

if TYPE_CHECKING:
    from environments.guarded_territory import GuardedTerritoryAdapter


class IPPOTrainer:
    def __init__(
        self,
        adapter: GuardedTerritoryAdapter,
        hidden_dim,
        lr,
        gamma,
        gae_lambda,
        clip_eps,
        value_coef,
        entropy_coef,
        device,
    ):
        self.device = device
        self.adapter = adapter
        self.num_agents = adapter.n_defenders

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        obs_dim = adapter.obs_dim
        action_dim = adapter.action_dim

        self.policy = IPPOPolicy(obs_dim, hidden_dim, action_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.critic = CriticNetwork(obs_dim, hidden_dim, device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr * 3)

        self.buffer = IPPORolloutBuffer(gamma=gamma, gae_lambda=gae_lambda, device=self.device)

        self._running_episode_return = 0.0
        self.metrics_history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "mean_bellman_error": [],
            "mean_episode_return": [],
            "mean_episode_rewards": [],
            "scout_reward": [],
            "interceptor_reward": [],
        }

        obs_batched, pos_batched = self.adapter.reset()
        self.current_obs = obs_batched.squeeze(0)
        self.current_positions = pos_batched.squeeze(0)

    def _safe_mean(self, values):
        return float(sum(values) / len(values)) if values else 0.0

    def collect_rollouts(self, num_steps):
        obs_tensor = self.current_obs
        positions = self.current_positions
        step_mean_rewards = []
        completed_episode_returns = []
        step_mean_scout_rew = []
        step_mean_inter_rew = []

        for _ in range(num_steps):
            actions, log_probs, entropy = self.policy.get_actions(obs=obs_tensor)
            values = self.critic(obs_tensor).detach().squeeze(-1)

            act_for_env = actions.unsqueeze(0)
            next_obs_b, rewards_b, dones_b, info, next_pos_b = self.adapter.step(act_for_env)

            next_obs = next_obs_b.squeeze(0)
            rewards_tensor = rewards_b.squeeze(0)
            next_positions = next_pos_b.squeeze(0)
            done_flag = dones_b.squeeze(0).item()

            n_s = self.adapter.n_scouts
            scout_rew = rewards_tensor[:n_s].mean().item()
            interceptor_rew = rewards_tensor[n_s:].mean().item()

            if done_flag:
                all_tagged = info.get("n_tagged", torch.tensor(0.0)).item() >= self.adapter.n_intruders
                all_breached = info.get("n_breached", torch.tensor(0.0)).item() >= self.adapter.n_zones
                true_done = all_tagged or all_breached
            else:
                true_done = False

            dones_for_gae = torch.full(
                (self.num_agents,),
                float(true_done),
                dtype=torch.float32,
                device=self.device,
            )

            self.buffer.add_timestep(
                obs=obs_tensor.detach(),
                actions=actions.detach(),
                rewards=rewards_tensor,
                dones=dones_for_gae,
                log_probs=log_probs.detach(),
                values=values,
            )

            step_mean_rewards.append(rewards_tensor.mean().item())
            step_mean_scout_rew.append(scout_rew)
            step_mean_inter_rew.append(interceptor_rew)
            self._running_episode_return += rewards_tensor.mean().item()

            if done_flag:
                completed_episode_returns.append(self._running_episode_return)
                self._running_episode_return = 0.0
                next_obs_b, next_pos_b = self.adapter.reset_env()
                next_obs = next_obs_b.squeeze(0)
                next_positions = next_pos_b.squeeze(0)

            obs_tensor = next_obs
            positions = next_positions

        self.current_obs = obs_tensor
        self.current_positions = positions

        rollout_metrics = {
            "mean_episode_return": self._safe_mean(completed_episode_returns)
            if completed_episode_returns
            else float(self._running_episode_return),
            "mean_episode_rewards": self._safe_mean(step_mean_rewards),
            "mean_scout_rewards": self._safe_mean(step_mean_scout_rew),
            "mean_inter_rewards": self._safe_mean(step_mean_inter_rew),
        }
        self.metrics_history["mean_episode_return"].append(rollout_metrics["mean_episode_return"])
        self.metrics_history["mean_episode_rewards"].append(rollout_metrics["mean_episode_rewards"])
        self.metrics_history["scout_reward"].append(rollout_metrics["mean_scout_rewards"])
        self.metrics_history["interceptor_reward"].append(rollout_metrics["mean_inter_rewards"])
        return obs_tensor, rollout_metrics

    def update(self, last_obs, num_actor_epochs=10, num_critic_epochs=5, B=64):
        with torch.no_grad():
            last_obs_tensor = last_obs.to(device=self.device, dtype=torch.float32)
            last_values = self.critic(last_obs_tensor).squeeze(-1)
        self.buffer.compute_advantages(last_values=last_values)

        policy_losses = []
        entropies = []
        value_losses = []
        bellman_errors = []

        for _ in range(num_actor_epochs):
            for obs, actions, old_log_probs, advantages, returns in self.buffer.get_batches(B):
                new_lp, entropy = self.policy.evaluate_actions(obs, actions)

                ratio = torch.exp(new_lp - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - self.entropy_coef * entropy_loss

                self.policy_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.policy_optim.step()

                policy_losses.append(policy_loss.item())
                entropies.append(entropy_loss.item())

        for _ in range(num_critic_epochs):
            for obs, actions, old_log_probs, advantages, returns in self.buffer.get_batches(B):
                batch_size, n_agents, obs_d = obs.shape
                flat_obs = obs.reshape(batch_size * n_agents, obs_d)
                flat_returns = returns.reshape(batch_size * n_agents)

                pred_values = self.critic(flat_obs).squeeze(-1)
                td_error = flat_returns - pred_values
                value_loss = F.mse_loss(pred_values, flat_returns)
                mean_bellman_error = td_error.abs().mean()

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()

                value_losses.append(value_loss.item())
                bellman_errors.append(mean_bellman_error.item())

        self.buffer.clear()
        update_metrics = {
            "policy_loss": self._safe_mean(policy_losses),
            "value_loss": self._safe_mean(value_losses),
            "entropy": self._safe_mean(entropies),
            "mean_bellman_error": self._safe_mean(bellman_errors),
        }
        for key in ["policy_loss", "value_loss", "entropy", "mean_bellman_error"]:
            self.metrics_history[key].append(update_metrics[key])
        return update_metrics
