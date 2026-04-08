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
        adv_s = (adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)
        adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

        #norm ret for val loss
        ret_s_mean, ret_s_std = ret_s.mean(), ret_s.std()
        ret_i_mean, ret_i_std = ret_i.mean(), ret_i.std()   
        ret_s_norm = (ret_s - ret_s_mean) / (ret_s_std + 1e-8)
        ret_i_norm = (ret_i - ret_i_mean) / (ret_i_std + 1e-8)

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
        flat_ret_s_norm = ret_s_norm.reshape(T*B)
        flat_ret_i_norm = ret_i_norm.reshape(T*B)
        flat_old_v_s = buffer.scout_values.reshape(T*B)
        flat_old_v_i = buffer.interc_values.reshape(T*B)

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


        

        






            




        