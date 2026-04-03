from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn

if TYPE_CHECKING:
    from environments.guarded_territory import GuardedTerritoryAdapter

from gnn.critic import CriticNetwork
from gnn.batch_rollout import GNNRolloutBuffer
from gnn.comm import CommPolicy

class GNNTrainer:
    def __init__(
        self,
        adapter: GuardedTerritoryAdapter,
        hidden_dim,
        F_feat,
        G_feat,
        K,
        lr,
        gamma,
        gae_lambda,
        clip_eps,
        value_coef,
        entropy_coef,
        device,
        r_comm,
    ):
        self.device = device
        self.adapter = adapter
        self.num_agents = adapter.n_defenders
        self.r_comm = r_comm

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        obs_dim = adapter.obs_dim
        action_dim = adapter.action_dim

        self.comm_policy = CommPolicy(
            obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim,
            F_feat=F_feat, G_feat=G_feat, K=K,
        ).to(self.device)
        self.comm_optim = Adam(self.comm_policy.parameters(), lr=lr)

        self.critic = CriticNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr*3)

        self.buffer = GNNRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda, device=self.device)

        self._running_episode_return = 0.0
        self.metrics_history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "mean_bellman_error": [],
            "explained_var": [],
            "mean_episode_return": [],
            "mean_episode_rewards": [],
            "scout_reward": [],
            "interceptor_reward": []
        }

        # Initial reset - squeeze out num_envs=1 dim to get (N, obs_dim)
        obs_batched, pos_batched = self.adapter.reset()
        self.current_obs = obs_batched.squeeze(0)       # (N, obs_dim)
        self.current_positions = pos_batched.squeeze(0)  # (N, 2)

    def _safe_mean(self, values):
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def collect_rollouts(self, num_steps):
        obs_tensor = self.current_obs           # (N, obs_dim)
        positions = self.current_positions       # (N, 2)
        step_mean_rewards = []
        completed_episode_returns = []
        step_mean_scout_rew = []
        step_mean_inter_rew = []

        for _ in range(num_steps):
            # Build adj from current positions: need (1, N, 2) for adapter
            S = self.adapter.build_adj(
                positions.unsqueeze(0), r_comm=self.r_comm
            ).squeeze(0)  # (N, N)

            # Policy forward: (N, obs_dim), (N, N) -> actions (N, 2)
            actions, log_probs, entropy = self.comm_policy.get_actions(
                obs=obs_tensor, S=S
            )

            # Critic: (N, obs_dim) -> (N,)
            values = self.critic(obs_tensor).detach().squeeze(-1)

            # Step env: adapter expects (num_envs, N, 2)
            act_for_env = actions.unsqueeze(0)  # (1, N, 2)
            next_obs_b, rewards_b, dones_b, info, next_pos_b = self.adapter.step(act_for_env)

            # Squeeze batch dim back out
            next_obs = next_obs_b.squeeze(0)         # (N, obs_dim)
            rewards_tensor = rewards_b.squeeze(0)     # (N,)
            next_positions = next_pos_b.squeeze(0)    # (N, 2)
            done_flag = dones_b.squeeze(0).item()     # scalar bool

            n_s = self.adapter.n_scouts
            scout_rew = rewards_tensor[:n_s].mean().item()
            interceptor_rew = rewards_tensor[n_s:].mean().item()
            if done_flag:
                if self.adapter.n_intruders == 0:
                    true_done = False
                else:
                    all_tagged = info.get("n_tagged", torch.tensor(0.0)).item() >= self.adapter.n_intruders
                    all_breached = info.get("n_breached", torch.tensor(0.0)).item() >= self.adapter.n_zones
                    true_done = all_tagged or all_breached
            else:
                true_done = False
            # Broadcast global done to all agents
            dones_for_gae = torch.full(
                (self.num_agents,), float(true_done),
                dtype=torch.float32, device=self.device,
            )

            # Store in buffer
            self.buffer.add_timestep(
                obs=obs_tensor.detach().unsqueeze(0),
                actions=actions.detach().unsqueeze(0),
                rewards=rewards_tensor.unsqueeze(0),
                dones=dones_for_gae.unsqueeze(0),
                log_probs=log_probs.detach().unsqueeze(0),
                values=values.unsqueeze(0),
                A=S.detach().unsqueeze(0),
            )

            # Metrics tracking
            step_mean_rewards.append(rewards_tensor.mean().item())
            step_mean_scout_rew.append(scout_rew)
            step_mean_inter_rew.append(interceptor_rew)
            self._running_episode_return += rewards_tensor.mean().item()

            if done_flag:
                completed_episode_returns.append(self._running_episode_return)
                self._running_episode_return = 0.0
                # VMAS auto-resets, so next_obs is already the new episode's obs
                next_obs_b, next_pos_b = self.adapter.reset_env()
                next_obs = next_obs_b.squeeze(0)
                next_positions = next_pos_b.squeeze(0)

            # Advance
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
            "mean_inter_rewards": self._safe_mean(step_mean_inter_rew)
        }
        self.metrics_history["mean_episode_return"].append(rollout_metrics["mean_episode_return"])
        self.metrics_history["mean_episode_rewards"].append(rollout_metrics["mean_episode_rewards"])
        self.metrics_history["scout_reward"].append(rollout_metrics["mean_scout_rewards"])
        self.metrics_history["interceptor_reward"].append(rollout_metrics["mean_inter_rewards"])
        return obs_tensor, rollout_metrics

    def update(self, last_obs, num_actor_epochs=15, num_critic_epochs=10, B=64):
        with torch.no_grad():
            last_obs_tensor = last_obs.to(device=self.device, dtype=torch.float32)
            last_values = self.critic(last_obs_tensor).squeeze(-1)
        self.buffer.compute_advantages(last_values=last_values)

        policy_losses = []
        entropies = []
        value_losses = []
        bellman_errors = []
        explained_vars = []

        #  actor update
        for _ in range(num_actor_epochs):
            for (obs, actions, old_log_probs, advantages, returns, A, _) in self.buffer.get_batches(B):
                new_lp, entropy, _ = self.comm_policy.evaluate_actions(obs, A, actions)

                ratio = torch.exp(new_lp - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - self.entropy_coef * entropy_loss

                self.comm_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.comm_policy.parameters(), max_norm=0.5)
                self.comm_optim.step()

                policy_losses.append(policy_loss.item())
                entropies.append(entropy_loss.item())

        # critic update (fewer epochs)
        for _ in range(num_critic_epochs):
            for (obs, actions, old_log_probs, advantages, returns, A, old_values) in self.buffer.get_batches(B):
                B_size, N, obs_d = obs.shape
                flat_obs = obs.reshape(B_size * N, obs_d)
                flat_returns = returns.reshape(B_size * N)

                pred_values = self.critic(flat_obs).squeeze(-1)
                td_error = flat_returns - pred_values
                value_loss = F.mse_loss(pred_values, flat_returns)
                mean_bellman_error = td_error.abs().mean()

                # Explained variance diagnostic
                ev = 1 - (flat_returns - pred_values).var() / (flat_returns.var() + 1e-8)

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()

                value_losses.append(value_loss.item())
                bellman_errors.append(mean_bellman_error.item())
                explained_vars.append(ev.item())

        self.buffer.clear()
        update_metrics = {
            "policy_loss": self._safe_mean(policy_losses),
            "value_loss": self._safe_mean(value_losses),
            "entropy": self._safe_mean(entropies),
            "mean_bellman_error": self._safe_mean(bellman_errors),
            "explained_var": self._safe_mean(explained_vars),
        }
        self.metrics_history["policy_loss"].append(update_metrics["policy_loss"])
        self.metrics_history["value_loss"].append(update_metrics["value_loss"])
        self.metrics_history["entropy"].append(update_metrics["entropy"])
        self.metrics_history["mean_bellman_error"].append(update_metrics["mean_bellman_error"])
        self.metrics_history["explained_var"].append(update_metrics["explained_var"])

        return update_metrics
