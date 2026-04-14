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

class GNNTrainerMultiEnv:
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
            total_timesteps,
            rollout_length,
    ):
        self.device = device
        self.adapter = adapter
        self.num_agents = adapter.n_defenders
        self.num_envs = adapter.num_envs    
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
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        if torch.__version__ >= "2.0":
            self.comm_policy = torch.compile(self.comm_policy)
            self.critic = torch.compile(self.critic)

        # lr annealing
        total_iter = total_timesteps // rollout_length
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.comm_optim, lr_lambda=lambda it: max(1 - it / total_iter, 0.05)
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optim, lr_lambda=lambda it: max(1-it/total_iter, 0.05)
        )

        self.buffer = GNNRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda, device=self.device)

        self._running_episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.metrics_history = {
            "policy_loss": [], "value_loss": [], "entropy":[],
            "mean_bellman_error":[], "mean_episode_return": [],
            "mean_episode_rewards": [], "scout_reward":[], "interceptor_reward":[],
            "explained_var":[], "comm_saliency": []
        }

        obs_batched, pos_batched = self.adapter.reset()
        self.current_obs = obs_batched
        self.current_positions = pos_batched

    
    def _safe_mean(self, values):
        return float(sum(values) / len(values)) if values else 0.0
    
    def collect_rollouts(self, num_steps):
        obs_tensor = self.current_obs
        positions = self.current_positions
        step_mean_rewards = []
        completed_episode_returns = []
        step_mean_scout_rew = []
        step_mean_inter_rew = []
        comm_saliency_accum = []

        n_s = self.adapter.n_scouts

        with torch.no_grad():
            for _ in range(num_steps):
                # build adj: (E, N, N)
                S = self.adapter.build_adj(positions, r_comm=self.r_comm)

                # policy forward: (E, N, obs_dim), (E, N, N) -> (E, N, 2)
                actions, log_probs, entropy = self.comm_policy.get_actions(obs=obs_tensor, S=S)

                # counterfactual communication saliency
                E, N = S.shape[0], S.shape[1]
                S_isolated = torch.eye(N, device=self.device).unsqueeze(0).expand(E, -1, -1)
                actions_nocm, _, _ = self.comm_policy.get_actions(obs=obs_tensor, S=S_isolated)
                comm_saliency = (actions - actions_nocm).norm(dim=-1).mean(dim=-1)
                comm_saliency_accum.append(comm_saliency.mean().item())

                # critic (E, N, obs_dim) -> (E, N)
                E, N, obs_d = obs_tensor.shape
                flat_obs = obs_tensor.reshape(E*N, obs_d)
                values = self.critic(flat_obs).squeeze(-1).reshape(E,N)


                # step env (E, N, 2)
                next_obs, rewards, dones, info, next_positions = self.adapter.step(actions)
                
                self._running_episode_returns += rewards.sum(dim=-1)
                # per type rewards average 
                scout_rew = rewards[:, :n_s].mean().item()
                interceptor_rew = rewards[:, n_s:].mean().item()

                # Truncation vs true done - per-env (modified to account for guided coverage env)
                if self.adapter.n_intruders == 0:
                    true_done_mask = dones.bool() if dones.dtype != torch.bool else dones
                else:
                    n_tagged = info.get("n_tagged", torch.zeros(E, device=self.device))
                    n_breached = info.get("n_breached", torch.zeros(E, device=self.device))

                    true_done_mask = (
                        dones
                        & ((n_tagged >= self.adapter.n_intruders)
                           | (n_breached >= self.adapter.n_zones))
                    )
                
                dones_for_gae = true_done_mask.float().unsqueeze(-1).expand(E,N)
                # Buffer: store (E, N, ...) tensors
                self.buffer.add_timestep(
                    obs=obs_tensor,
                    actions=actions,
                    rewards=rewards,
                    dones=dones_for_gae,
                    log_probs=log_probs,
                    values=values,
                    A=S,
                )

                # Metrics
                step_mean_rewards.append(rewards.mean().item())
                step_mean_scout_rew.append(scout_rew)
                step_mean_inter_rew.append(interceptor_rew)

                # Track per-env episode returns
                completed_mask = dones
                if completed_mask.any():
                    for e in completed_mask.nonzero(as_tuple=True)[0].tolist():
                        completed_episode_returns.append(self._running_episode_returns[e].item())
                    self._running_episode_returns[completed_mask] = 0.0

                # VMAS auto-resets done envs, so next_obs already has fresh obs for those envs
                obs_tensor = next_obs
                positions = next_positions

            self.current_obs = obs_tensor
            self.current_positions = positions

            rollout_metrics = {
                "mean_episode_return": self._safe_mean(completed_episode_returns)
                    if completed_episode_returns else float(self._running_episode_returns.mean().item()),
                "mean_episode_rewards": self._safe_mean(step_mean_rewards),
                "mean_scout_rewards": self._safe_mean(step_mean_scout_rew),
                "mean_inter_rewards": self._safe_mean(step_mean_inter_rew),
                "comm_saliency": self._safe_mean(comm_saliency_accum),
            }
            self.metrics_history["mean_episode_return"].append(rollout_metrics["mean_episode_return"])
            self.metrics_history["mean_episode_rewards"].append(rollout_metrics["mean_episode_rewards"])
            self.metrics_history["scout_reward"].append(rollout_metrics["mean_scout_rewards"])
            self.metrics_history["interceptor_reward"].append(rollout_metrics["mean_inter_rewards"])
            self.metrics_history["comm_saliency"].append(rollout_metrics["comm_saliency"])
        return obs_tensor, rollout_metrics
    

    def update(self, last_obs, num_actor_epochs=10, num_critic_epochs=10, B=64):
        with torch.no_grad():
            E, N, obs_d = last_obs.shape
            flat_obs = last_obs.to(device=self.device, dtype=torch.float32).reshape(E * N, obs_d)
            last_values = self.critic(flat_obs).squeeze(-1).reshape(E, N)
        self.buffer.compute_advantages(last_values=last_values)

        policy_losses, entropies, value_losses, bellman_errors, explained_vars = [], [], [], [], []

        # Actor update
        for _ in range(num_actor_epochs):
            for batch in self.buffer.get_batches(B, connectivity_bias=True):
                obs, actions, old_log_probs, advantages, returns, A, old_values = batch
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
                with torch.no_grad():
                    self.comm_policy.log_std.clamp_(min=-1.5, max=0.5)
                policy_losses.append(policy_loss.item())
                entropies.append(entropy_loss.item())

        # Critic update
        for _ in range(num_critic_epochs):
            for batch in self.buffer.get_batches(B, connectivity_bias=True):
                obs, actions, old_log_probs, advantages, returns, A, old_values = batch
                B_size, N, obs_d = obs.shape
                flat_obs = obs.reshape(B_size * N, obs_d)
                flat_returns = returns.reshape(B_size * N)
                flat_old_values = old_values.reshape(B_size * N)

                pred_values = self.critic(flat_obs).squeeze(-1)

                # value loss clipping
                values_clipped = flat_old_values + torch.clamp(pred_values - flat_old_values, -self.clip_eps, self.clip_eps)
                value_loss_unclipped = F.mse_loss(pred_values, flat_returns, reduction='none')
                value_loss_clipped = F.mse_loss(values_clipped, flat_returns, reduction='none')
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                td_error = flat_returns - pred_values
                mean_bellman_error = td_error.abs().mean()

                with torch.no_grad():
                    ev = 1 - (flat_returns - pred_values).var() / (flat_returns.var() + 1e-8)
                    explained_vars.append(ev.item())

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()

                value_losses.append(value_loss.item())
                bellman_errors.append(mean_bellman_error.item())

        self.buffer.clear()

        # Step LR schedulers
        self.policy_scheduler.step()
        self.critic_scheduler.step()

        update_metrics = {
            "policy_loss": self._safe_mean(policy_losses),
            "value_loss": self._safe_mean(value_losses),
            "entropy": self._safe_mean(entropies),
            "mean_bellman_error": self._safe_mean(bellman_errors),
            "explained_var": self._safe_mean(explained_vars)
        }
        for k in update_metrics:
            self.metrics_history[k].append(update_metrics[k])
        return update_metrics
