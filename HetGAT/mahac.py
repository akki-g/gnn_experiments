import torch
import torch.nn as nn
import torch.nn.functional as F

from HetGAT.hetnetPolicy import HetNetPolicy
from HetGAT.critic import HetNetCritic
from HetGAT.rollout import MAHACBuffer
from HetGAT.utils import compute_gae, build_ssn_input


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
            critic_epochs: int = 1,
            gamma: float = 0.99,
            lam: float = 0.95,
            clip_eps: float = 0.2,
            entropy_coef: float = 0.01, 
            value_coef: float = 0.25, # was 0.5
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

        self.critic_epochs = critic_epochs
        self.actor_optim = torch.optim.Adam(
            policy.parameters(), lr=lr_actor
        )
        self.critic_optim = torch.optim.Adam(
            critic.parameters(), lr=lr_critic
        )

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
        
        changed to be batched
        """

        T = buffer.T
        B = buffer.B
        n_s = buffer.n_s
        n_i = buffer.n_i
        device = buffer.device

        # gae computation unchanged
        with torch.no_grad():
            if self.critic.mode != 'centralized':
                adv_s , ret_s = compute_gae(
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

            raw_adv_mean_s = adv_s.mean().item()
            raw_adv_std_s = adv_s.std().item()
            raw_adv_mean_i = adv_i.mean().item()
            raw_adv_std_i = adv_i.std().item()

            old_log_probs_s = buffer.scout_log_probs.detach()
            old_log_probs_i = buffer.interc_log_probs.detach()

        #norm adv
        adv_s = (adv_s - adv_s.mean()) / adv_s.std().clamp(min=1e-4)  # FIX-P3-B10: floor prevents amplification
        adv_i = (adv_i - adv_i.mean()) / adv_i.std().clamp(min=1e-4)  # FIX-P3-B10

        # FIX-2: do NOT normalize returns — train critic on raw returns (CleanRL/SpinUp standard)
        # Simple batch normalization creates scale mismatch between critic training and EV evaluation

        # preflatten buffer data for batched forward pass
        flat_obs_s = buffer.scout_obs.reshape(T*B, n_s, -1)
        flat_state_s = buffer.scout_states.reshape(T*B, n_s, -1)
        flat_obs_i = buffer.interc_obs.reshape(T*B, n_i, -1)
        flat_state_i = buffer.interc_states.reshape(T*B, n_i, -1)
        flat_positions = buffer.positions.reshape(T*B, n_s+n_i, -1)
        flat_ssn = buffer.ssn_inputs.reshape(T*B, -1)
        flat_actions_s = buffer.scout_actions.reshape(T*B, n_s, -1)
        flat_actions_i = buffer.interc_actions.reshape(T*B, n_i, -1)


        flat_hidden_s = {
            'obs_h': buffer.scout_hidden[:, 0].reshape(T*B*n_s, -1),
            'obs_c': buffer.scout_hidden[:, 1].reshape(T*B*n_s, -1),
            'state_h': buffer.scout_hidden[:, 2].reshape(T*B*n_s, -1),
            'state_c': buffer.scout_hidden[:, 3].reshape(T*B*n_s, -1)
        }
        flat_hidden_i = {
            'obs_h': buffer.interc_hidden[:, 0].reshape(T*B*n_i, -1),
            'obs_c': buffer.interc_hidden[:, 1].reshape(T*B*n_i, -1),
            'state_h': buffer.interc_hidden[:, 2].reshape(T*B*n_i, -1),
            'state_c': buffer.interc_hidden[:, 3].reshape(T*B*n_i, -1)
        }

        flat_adv_s = adv_s.reshape(T*B)
        flat_adv_i = adv_i.reshape(T*B)
        flat_old_lp_s = old_log_probs_s.reshape(T*B, n_s)
        flat_old_lp_i = old_log_probs_i.reshape(T*B, n_i)
        # FIX-2: use raw (unnormalized) returns as critic targets
        flat_ret_s = ret_s.reshape(T*B)
        flat_ret_i = ret_i.reshape(T*B)
        # flat_old_v_s / flat_old_v_i removed — value clipping removed in FIX-6

        #metric trackers
        total_policy_loss_s = 0.0
        total_policy_loss_i = 0.0
        total_value_loss = 0.0
        total_entropy_s = 0.0
        total_entropy_i = 0.0
        total_ratio_s = 0.0
        total_ratio_i = 0.0
        n_actor_updates = 0
        n_critic_updates = 0
        # FIX-P2-DIAG: track log-prob difference per epoch (epoch 0 should be ~0, epoch 1+ nonzero)
        lp_diff_epoch0_s = 0.0
        lp_diff_epoch0_i = 0.0
        lp_diff_final_s = 0.0
        lp_diff_final_i = 0.0
        total_actor_grad_norm = 0.0   # FIX-P4-B4: pre-clip grad norm for entropy_coef diagnostics
        total_ent_loss_contrib = 0.0  # FIX-P4-B4: entropy loss contribution tracking

        #actor epochs: one forwards pass per epoch instead of T
        for epoch in range(self.ppo_epochs):
            scout_dist, interc_dist, h_ssn, _, _ = self.policy(
                flat_obs_s, flat_state_s, flat_obs_i, flat_state_i,
                flat_positions, flat_ssn,
                flat_hidden_s, flat_hidden_i,
                n_s, n_i
            )
            
            #new log probs
            new_lp_s = scout_dist.log_prob(flat_actions_s).sum(dim=-1)
            new_lp_i = interc_dist.log_prob(flat_actions_i).sum(dim=-1)

            #ppo ratios
            ratio_s = torch.exp(new_lp_s - flat_old_lp_s)
            ratio_i = torch.exp(new_lp_i - flat_old_lp_i)

            # FIX-P2-DIAG: epoch-0 lp_diff should be ~0; nonzero after first optimizer step
            with torch.no_grad():
                _lp_diff_s = (new_lp_s - flat_old_lp_s).abs().mean().item()
                _lp_diff_i = (new_lp_i - flat_old_lp_i).abs().mean().item()
                if epoch == 0:
                    lp_diff_epoch0_s = _lp_diff_s
                    lp_diff_epoch0_i = _lp_diff_i
                lp_diff_final_s = _lp_diff_s
                lp_diff_final_i = _lp_diff_i

            #expand per class adv to per agent
            adv_s_exp = flat_adv_s.unsqueeze(-1).expand_as(ratio_s)
            adv_i_exp = flat_adv_i.unsqueeze(-1).expand_as(ratio_i)
        
            #clip surrogates
            surr1_s = ratio_s * adv_s_exp
            surr2_s = torch.clamp(ratio_s, 1 - self.clip_eps, 1 + self.clip_eps) * adv_s_exp
            policy_loss_s = -torch.min(surr1_s, surr2_s).mean()

            surr1_i = ratio_i * adv_i_exp
            surr2_i = torch.clamp(ratio_i, 1-self.clip_eps, 1+self.clip_eps) * adv_i_exp
            policy_loss_i = -torch.min(surr1_i, surr2_i).mean()

            #entropy
            entropy_s = scout_dist.entropy().sum(dim=-1).mean()
            entropy_i = interc_dist.entropy().sum(dim=-1).mean()

            #combined actor loss
            actor_loss = (
                policy_loss_s + policy_loss_i
                - self.entropy_coef * (entropy_s + entropy_i)
            )

            self.actor_optim.zero_grad()
            actor_loss.backward()

            # FIX-P4-B4: log pre-clip actor grad norm for entropy_coef diagnostics
            _gnorm = sum(
                p.grad.norm() ** 2 for p in self.policy.parameters() if p.grad is not None
            ).sqrt().item()
            total_actor_grad_norm += _gnorm
            total_ent_loss_contrib += abs(self.entropy_coef * (entropy_s.item() + entropy_i.item()))

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.actor_optim.step()

            # clamp log std (min and max)
            with torch.no_grad():
                self.policy.log_std_scout.clamp_(min=self.log_std_min, max=0.5)
                self.policy.log_std_interc.clamp_(min=self.log_std_min, max=0.5)
            
            total_policy_loss_s += policy_loss_s.item()
            total_policy_loss_i += policy_loss_i.item()
            total_entropy_s += entropy_s.item()
            total_entropy_i += entropy_i.item()
            total_ratio_s += ratio_s.mean().item()
            total_ratio_i += ratio_i.mean().item()
            n_actor_updates += 1

        
        #critic epochs 
        for epoch in range(self.critic_epochs):
            with torch.no_grad():
                _, _, h_ssn, _, _ = self.policy(
                    flat_obs_s, flat_state_s, flat_obs_i, flat_state_i,
                    flat_positions, flat_ssn, 
                    flat_hidden_s, flat_hidden_i,
                    n_s, n_i
                )

            values = self.critic(h_ssn)
            v_s = values['scout'].squeeze(-1)
            v_i = values['interc'].squeeze(-1)

            # FIX-2+6: simple MSE against raw returns — no value clipping
            # Value clipping at ±0.2 would prevent the critic from learning when raw returns
            # have a large scale (e.g. multi-step discounted returns). SpinUp uses plain MSE.
            vl_s = F.mse_loss(v_s, flat_ret_s)
            vl_i = F.mse_loss(v_i, flat_ret_i)

            critic_loss = self.value_coef * (vl_s + vl_i)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()

            total_value_loss += (vl_s + vl_i).item()
            n_critic_updates += 1


        #post update ev
        with torch.no_grad():
            _, _, h_ssn_post, _, _ = self.policy(
                flat_obs_s, flat_state_s, flat_obs_i, flat_state_i,
                flat_positions, flat_ssn,
                flat_hidden_s, flat_hidden_i, 
                n_s, n_i
            )

            post_vals = self.critic(h_ssn_post)
            post_v_s = post_vals['scout'].squeeze(-1)
            post_v_i = post_vals['interc'].squeeze(-1)

            # flat_ret_s, flat_ret_i already defined above (FIX-2)
            flat_vals_s = buffer.scout_values.reshape(T*B)
            flat_vals_i = buffer.interc_values.reshape(T*B)

            ev_scout_post = (1.0 - (flat_ret_s - post_v_s).var() / (flat_ret_s.var() + 1e-8)).item()
            ev_interc_post = (1.0 - (flat_ret_i - post_v_i).var() / (flat_ret_i.var() + 1e-8)).item()   
            ev_scout = (1.0 - (flat_ret_s - v_s).var() / (flat_ret_s.var() + 1e-8)).item()
            ev_interc = (1.0 - (flat_ret_i - v_i).var() / (flat_ret_i.var() + 1e-8)).item()
            msbe_scout = F.mse_loss(flat_vals_s, flat_ret_s).item()
            msbe_interc = F.mse_loss(flat_vals_i, flat_ret_i).item()

         
        metrics = {
            'policy_loss_scout': total_policy_loss_s / n_actor_updates,
            'policy_loss_interc': total_policy_loss_i / n_actor_updates,
            'value_loss': total_value_loss / n_critic_updates,
            'entropy_scout': total_entropy_s / n_actor_updates,
            'entropy_interc': total_entropy_i / n_actor_updates,
            'log_std_scout': self.policy.log_std_scout.data.mean().item(),
            'log_std_interc': self.policy.log_std_interc.data.mean().item(),
            'ratio_scout_mean': total_ratio_s / n_actor_updates,
            'ratio_interc_mean': total_ratio_i / n_actor_updates,
            'ev_scout': ev_scout,
            'ev_interc': ev_interc,
            'ev_scout_post': ev_scout_post,
            'ev_interc_post': ev_interc_post,
            'msbe_scout': msbe_scout,
            'msbe_interc': msbe_interc,
            'advantage_mean_scout': raw_adv_mean_s,
            'advantage_std_scout': raw_adv_std_s,
            'advantage_mean_interc': raw_adv_mean_i,
            'advantage_std_interc': raw_adv_std_i,
            'lp_diff_epoch0_scout': lp_diff_epoch0_s,   # FIX-P2-DIAG: should be ~0 at epoch 0
            'lp_diff_epoch0_interc': lp_diff_epoch0_i,
            'lp_diff_final_scout': lp_diff_final_s,     # FIX-P2-DIAG: nonzero after optimizer step
            'lp_diff_final_interc': lp_diff_final_i,
            'actor_grad_norm_preclip': total_actor_grad_norm / max(n_actor_updates, 1),  # FIX-P4-B4
            'ent_loss_contribution': total_ent_loss_contrib / max(n_actor_updates, 1),   # FIX-P4-B4
        }
        return metrics

        






            




        