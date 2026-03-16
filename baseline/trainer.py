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
    """
    Multi-environment IPPO trainer.

    Works with num_envs > 1. All rollout tensors are (E, N, ...)
    where E = num_envs, N = n_defenders. Mirrors GNNTrainerMultiEnv
    exactly, minus the graph communication layer and adjacency matrix.
    """

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
        self.device      = device
        self.adapter     = adapter
        self.num_agents  = adapter.n_defenders

        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef

        obs_dim    = adapter.obs_dim
        action_dim = adapter.action_dim

        self.policy       = IPPOPolicy(obs_dim, hidden_dim, action_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.critic       = CriticNetwork(obs_dim, hidden_dim, device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr * 3)

        self.buffer = IPPORolloutBuffer(gamma=gamma, gae_lambda=gae_lambda, device=self.device)

        # Per-env running return tracker — shape (E,)
        self._running_episode_returns = torch.zeros(adapter.num_envs, device=self.device)

        self.metrics_history = {
            "policy_loss":        [],
            "value_loss":         [],
            "entropy":            [],
            "mean_bellman_error": [],
            "mean_episode_return":  [],
            "mean_episode_rewards": [],
            "scout_reward":         [],
            "interceptor_reward":   [],
        }

        # Reset returns (E, N, obs_dim) — keep env dim, no squeeze
        obs_batched, pos_batched = self.adapter.reset()
        self.current_obs       = obs_batched   # (E, N, obs_dim)
        self.current_positions = pos_batched   # (E, N, 2)

    # ─────────────────────────────────────────────────────────────
    def _safe_mean(self, values):
        return float(sum(values) / len(values)) if values else 0.0

    # ─────────────────────────────────────────────────────────────
    def collect_rollouts(self, num_steps):
        obs_tensor = self.current_obs       # (E, N, obs_dim)
        positions  = self.current_positions # (E, N, 2)

        step_mean_rewards      = []
        completed_episode_returns = []
        step_mean_scout_rew    = []
        step_mean_inter_rew    = []

        n_s = self.adapter.n_scouts

        with torch.no_grad():
            for _ in range(num_steps):
                E, N, obs_d = obs_tensor.shape

                # ── Policy forward: (E, N, obs_dim) → (E, N, act_dim) ──
                actions, log_probs, entropy = self.policy.get_actions(obs=obs_tensor)

                # ── Critic: flatten → (E*N, obs_dim) → reshape → (E, N) ──
                flat_obs = obs_tensor.reshape(E * N, obs_d)
                values   = self.critic(flat_obs).squeeze(-1).reshape(E, N)

                # ── Env step: actions (E, N, 2) passed directly ──
                next_obs, rewards, dones, info, next_positions = self.adapter.step(actions)
                # rewards: (E, N),  dones: (E,) bool

                scout_rew      = rewards[:, :n_s].mean().item()
                interceptor_rew = rewards[:, n_s:].mean().item()

                # ── Vectorized done detection — no CPU syncs ──
                n_tagged   = info.get("n_tagged",   torch.zeros(E, device=self.device))
                n_breached = info.get("n_breached", torch.zeros(E, device=self.device))
                true_done_mask = (
                    dones
                    & ((n_tagged >= self.adapter.n_intruders)
                       | (n_breached >= self.adapter.n_zones))
                )  # (E,) bool

                # Broadcast to (E, N) for GAE
                dones_for_gae = true_done_mask.float().unsqueeze(-1).expand(E, N)

                # ── Buffer store ──
                self.buffer.add_timestep(
                    obs=obs_tensor,
                    actions=actions,
                    rewards=rewards,
                    dones=dones_for_gae,
                    log_probs=log_probs,
                    values=values,
                )

                # ── Metrics ──
                step_mean_rewards.append(rewards.mean().item())
                step_mean_scout_rew.append(scout_rew)
                step_mean_inter_rew.append(interceptor_rew)

                # ── Episode return tracking (vectorized, minimal syncs) ──
                self._running_episode_returns += rewards.mean(dim=-1)  # (E,)
                if true_done_mask.any():
                    for e in true_done_mask.nonzero(as_tuple=True)[0].tolist():
                        completed_episode_returns.append(
                            self._running_episode_returns[e].item()
                        )
                    self._running_episode_returns[true_done_mask] = 0.0

                obs_tensor = next_obs
                positions  = next_positions

        self.current_obs       = obs_tensor
        self.current_positions = positions

        rollout_metrics = {
            "mean_episode_return": (
                self._safe_mean(completed_episode_returns)
                if completed_episode_returns
                else float(self._running_episode_returns.mean().item())
            ),
            "mean_episode_rewards": self._safe_mean(step_mean_rewards),
            "mean_scout_rewards":   self._safe_mean(step_mean_scout_rew),
            "mean_inter_rewards":   self._safe_mean(step_mean_inter_rew),
        }

        self.metrics_history["mean_episode_return"].append(rollout_metrics["mean_episode_return"])
        self.metrics_history["mean_episode_rewards"].append(rollout_metrics["mean_episode_rewards"])
        self.metrics_history["scout_reward"].append(rollout_metrics["mean_scout_rewards"])
        self.metrics_history["interceptor_reward"].append(rollout_metrics["mean_inter_rewards"])

        return obs_tensor, rollout_metrics

    # ─────────────────────────────────────────────────────────────
    def update(self, last_obs, num_actor_epochs=10, num_critic_epochs=5, B=64):
        # Compute last values for GAE bootstrap
        with torch.no_grad():
            E, N, obs_d = last_obs.shape
            flat_obs    = last_obs.to(device=self.device, dtype=torch.float32).reshape(E * N, obs_d)
            last_values = self.critic(flat_obs).squeeze(-1).reshape(E, N)

        self.buffer.compute_advantages(last_values=last_values)

        policy_losses  = []
        entropies      = []
        value_losses   = []
        bellman_errors = []
        explained_vars = []

        # ── Actor update ──────────────────────────────────────────
        for _ in range(num_actor_epochs):
            for obs, actions, old_log_probs, advantages, returns in self.buffer.get_batches(B):
                # obs: (B, N, obs_dim), actions: (B, N, act_dim)
                new_lp, entropy = self.policy.evaluate_actions(obs, actions)

                ratio  = torch.exp(new_lp - old_log_probs)
                surr1  = ratio * advantages
                surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

                policy_loss  = -torch.min(surr1, surr2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - self.entropy_coef * entropy_loss

                self.policy_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.policy_optim.step()

                policy_losses.append(policy_loss.item())
                entropies.append(entropy_loss.item())

        # ── Critic update ─────────────────────────────────────────
        for _ in range(num_critic_epochs):
            for obs, actions, old_log_probs, advantages, returns in self.buffer.get_batches(B):
                batch_size, n_agents, obs_d = obs.shape
                flat_obs_c    = obs.reshape(batch_size * n_agents, obs_d)
                flat_returns  = returns.reshape(batch_size * n_agents)

                pred_values = self.critic(flat_obs_c).squeeze(-1)
                td_error    = flat_returns - pred_values
                value_loss  = F.mse_loss(pred_values, flat_returns)
                mean_bellman_error = td_error.abs().mean()

                # Explained variance diagnostic
                var_returns = flat_returns.var() + 1e-8
                exp_var = 1.0 - td_error.var() / var_returns

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()

                value_losses.append(value_loss.item())
                bellman_errors.append(mean_bellman_error.item())
                explained_vars.append(exp_var.item())

        self.buffer.clear()

        update_metrics = {
            "policy_loss":        self._safe_mean(policy_losses),
            "value_loss":         self._safe_mean(value_losses),
            "entropy":            self._safe_mean(entropies),
            "mean_bellman_error": self._safe_mean(bellman_errors),
            "explained_var":      self._safe_mean(explained_vars),
        }
        for key in ["policy_loss", "value_loss", "entropy", "mean_bellman_error"]:
            self.metrics_history[key].append(update_metrics[key])

        return update_metrics