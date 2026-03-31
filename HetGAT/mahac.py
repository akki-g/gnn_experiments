import torch
import torch.nn as nn
import torch.nn.functional as F

from hetnetPolicy import HetNetPolicy
from critic import HetNetCritic
from rollout import MAHACBuffer
from utils import compute_gae, build_ssn_input


class MAHAC:
    """
    Multi Agent Heterogenus Actor Critic

    PPO style updates with per-class policies sharing a HetGAT backbone.
    One optimize over all params - both class losses contribute gradients
    to shared. inter class edge weights in the HetGAT layers
    """

    def __init__(
            self,
            policy: HetNetPolicy,
            critic: HetNetCritic, 
            lr_actor: float = 3e-4,
            lr_critic: float = 1e-3,
            gamma: float = 0.99,
            lam: float = 0.95,
            clip_eps: float = 0.2,
            entropy_coef: float = 0.01, 
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            ppo_epochs: int = 4,
            log_std_min: float = -1.5 # clamp to prevent entropy collapse
            ):
        
        self.policy = policy
        self.critic = critic
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.log_std_min = log_std_min

        # single optim for all gradients
        # hetGAT backbone params get gradients from BOTH class losses
        self.optimizer = torch.optim.Adam([
            {'params': policy.parameters(), 'lr': lr_actor},
            {'params': critic.parameters(), 'lr': lr_critic}
        ])

    def update(
            self,
            buffer: MAHACBuffer,
            bootstrap_values: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """
        run ppo update on collected trajectories

        uses stored LSTM hidden states (truncated BPTT, chunk=1) to 
        recompute forward passes for fresh log_probs and value estimates

        args:
            buffer: filled MAHACBuffer from collect_rollouts
            bootstrap_values: dict with 'scout' and 'interc' V(s_t+1) (or just 'all' for centralized critic)
        
        returns:
            metrics: dict of training metrics
        """
        T = buffer.T
        B = buffer.B
        device = buffer.device

        # compute gae with stored value estimates
        with torch.no_grad():
            if self.critic.mode != 'centralized':
                adv_s, ret_s = compute_gae(
                    buffer.rewards, buffer.scout_values, buffer.dones,
                    bootstrap_values['scout'], self.gamma, self.lam
                )
                adv_i, ret_i = compute_gae(
                    buffer.rewards, buffer.interc_values, buffer.dones,
                    bootstrap_values['interc'], self.gamma, self.lam
                )
            else:
                all_values = torch.cat([buffer.scout_values, buffer.interc_values], dim=1)
                adv_t, ret_t = compute_gae(
                    buffer.rewards, all_values, buffer.dones,
                    bootstrap_values['all'], self.gamma, self.lam
                )

            # capture raw advantage statistics before normalization
            raw_adv_mean_s = adv_s.mean().item()
            raw_adv_std_s = adv_s.std().item()
            raw_adv_mean_i = adv_i.mean().item()
            raw_adv_std_i = adv_i.std().item()

            # old log probs for ratio
            old_log_probs_s = buffer.scout_log_probs.detach()
            old_log_probs_i = buffer.interc_log_probs.detach()

        #normalize advantages
        adv_s = (adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)
        adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

        # metrics to track
        total_policy_loss_s = 0.0
        total_policy_loss_i = 0.0
        total_value_loss = 0.0
        total_entropy_s = 0.0
        total_entropy_i = 0.0
        total_ratio_s = 0.0
        total_ratio_i = 0.0
        n_updates = 0

        #ppo epochs
        for epoch in range(self.ppo_epochs):
            epoch_policy_loss_s = 0.0
            epoch_policy_loss_i = 0.0
            epoch_value_loss = 0.0
            epoch_entropy_s = 0.0
            epoch_entropy_i = 0.0
            epoch_ratio_s = 0.0
            epoch_ratio_i = 0.0

            for t in range(T):
                # load stored hidden states from buffer
                hidden_s = buffer.get_hidden_scout(t)
                hidden_i = buffer.get_hidden_interc(t)

                # fresh forward pass
                scout_dist, interc_dist, h_ssn, _, _ = self.policy(
                    buffer.scout_obs[t],
                    buffer.scout_states[t],
                    buffer.interc_obs[t],
                    buffer.interc_states[t],
                    buffer.positions[t],
                    buffer.ssn_inputs[t],
                    hidden_s, hidden_i,
                    buffer.n_s, buffer.n_i
                )

                # get fresh log probs (buffer stores pre-tanh raw actions)
                _, new_log_prob_s, _ = self.policy.squash_action(scout_dist, raw_action=buffer.scout_actions[t])
                _, new_log_prob_i, _ = self.policy.squash_action(interc_dist, raw_action=buffer.interc_actions[t])

                # fresh value estimates
                values = self.critic(h_ssn)
                v_s = values['scout'].squeeze(-1)
                v_i = values['interc'].squeeze(-1)

                #scout ppo loss
                ratio_s = torch.exp(new_log_prob_s - old_log_probs_s[t])
                #expand per class adv to per-agent
                adv_s_t = adv_s[t].unsqueeze(-1).expand_as(ratio_s)

                surr1_s = ratio_s * adv_s_t
                surr2_s = torch.clamp(
                    ratio_s, 1-self.clip_eps, 1+self.clip_eps
                ) * adv_s_t
                policy_loss_s = -torch.min(surr1_s, surr2_s).mean()

                # interceptor ppo loss
                ratio_i = torch.exp(new_log_prob_i - old_log_probs_i[t])
                adv_i_t = adv_i[t].unsqueeze(-1).expand_as(ratio_i)
                surr1_i = ratio_i * adv_i_t
                surr2_i = torch.clamp(ratio_i, 1-self.clip_eps, 1+self.clip_eps)*adv_i_t
                policy_loss_i = -torch.min(surr1_i, surr2_i).mean()

                # critic value loss for both classes
                value_loss = (
                    F.mse_loss(v_s, ret_s[t])
                    + F.mse_loss(v_i, ret_i[t])
                )

                # entropy bonus for both classes
                entropy_s = scout_dist.entropy().sum(dim=-1).mean()
                entropy_i = interc_dist.entropy().sum(dim=-1).mean()

                # accumulate over time
                epoch_policy_loss_s += policy_loss_s
                epoch_policy_loss_i += policy_loss_i
                epoch_value_loss += value_loss
                epoch_entropy_s += entropy_s.item()
                epoch_entropy_i += entropy_i.item()
                epoch_ratio_s += ratio_s.mean().item()
                epoch_ratio_i += ratio_i.mean().item()

            # average over timesteps and compute total loss
            total_loss = (
                (epoch_policy_loss_s + epoch_policy_loss_i) / T
                + self.value_coef * epoch_value_loss / T
                - self.entropy_coef * (entropy_s + entropy_i) / T
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            # clamp log_std to prevent entropy collapse
            with torch.no_grad():
                self.policy.log_std_scout.clamp_(min=self.log_std_min)
                self.policy.log_std_interc.clamp_(min=self.log_std_min)

            # accumulate metrics
            total_policy_loss_s += epoch_policy_loss_s.item() / T
            total_policy_loss_i += epoch_policy_loss_i.item() / T
            total_value_loss += epoch_value_loss.item() / T
            total_entropy_s += epoch_entropy_s / T
            total_entropy_i += epoch_entropy_i / T
            total_ratio_s += epoch_ratio_s / T
            total_ratio_i += epoch_ratio_i / T
            n_updates += 1

        # compute explained variance and MSBE once per update (outside no_grad -- ret_s already detached)
        with torch.no_grad():
            values_s_flat = buffer.scout_values.flatten()
            values_i_flat = buffer.interc_values.flatten()
            ret_s_flat = ret_s.flatten()
            ret_i_flat = ret_i.flatten()

            # explained variance: 1 - Var(returns - values) / Var(returns)
            var_ret_s = ret_s_flat.var()
            var_ret_i = ret_i_flat.var()
            ev_scout = (1.0 - (ret_s_flat - values_s_flat).var() / (var_ret_s + 1e-8)).item()
            ev_interc = (1.0 - (ret_i_flat - values_i_flat).var() / (var_ret_i + 1e-8)).item()

            # mean squared Bellman error
            msbe_scout = F.mse_loss(values_s_flat, ret_s_flat).item()
            msbe_interc = F.mse_loss(values_i_flat, ret_i_flat).item()

        metrics = {
            'policy_loss_scout': total_policy_loss_s / n_updates,
            'policy_loss_interc': total_policy_loss_i / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_scout': total_entropy_s / n_updates,
            'entropy_interc': total_entropy_i / n_updates,
            'log_std_scout': self.policy.log_std_scout.data.mean().item(),
            'log_std_interc': self.policy.log_std_interc.data.mean().item(),
            'ratio_scout_mean': total_ratio_s / n_updates,
            'ratio_interc_mean': total_ratio_i / n_updates,
            'ev_scout': ev_scout,
            'ev_interc': ev_interc,
            'msbe_scout': msbe_scout,
            'msbe_interc': msbe_interc,
            'advantage_mean_scout': raw_adv_mean_s,
            'advantage_std_scout': raw_adv_std_s,
            'advantage_mean_interc': raw_adv_mean_i,
            'advantage_std_interc': raw_adv_std_i,
        }
        return metrics






            




        