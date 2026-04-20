from environments.guarded_territory import GuardedTerritoryAdapter
from HetGAT.hetnetPolicy import HetNetPolicy
from HetGAT.rollout import MAHACBuffer
from HetGAT.critic import HetNetCritic

import torch


def build_ssn_input(
        n_scouts: int,
        n_interc: int,
        n_intrud: int,
        world_size: int,
        t_norm: int, 
        B: int,
        device: torch.device
) -> torch.Tensor:
    """
    Construct SSN feature vector, batched.

    Packs env hyperparams into a fixed sized tensor
    as the init node feat for ssn

    args:
        n_scout: number of scout agents 
        n_interc: number of intercept agents
        n_intruders: number of intruder agents 
        world_size: env world half size
        t_norm: norm time step 
        B: batch size 
        device: torch device
    """

    raw = torch.tensor(
        [float(n_scouts), float(n_interc), float(n_intrud), world_size, t_norm],
        dtype=torch.float32,
        device=device
    )

    return raw.unsqueeze(0).expand(B, -1)

def combine_actions(
        scout_actions: torch.Tensor,
        interc_actions: torch.Tensor
) -> list[torch.Tensor]:
    """
    combine per-class actions into the format env_adapter.step() expects

    scouts first then interceptors
    VMAS -> expects a list of per-agent action tensors

    args:
        scout_actions: (B, n_s, action_dim)
        interc_actions: (B, n_i, action_dim)

    returns:
    list of (B, action_dim) tensors, one per defender
    """

    return torch.cat([scout_actions, interc_actions], dim=1)


def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_value: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    compute generalized adv estimation
    """

    T, B = rewards.shape    
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    next_value = bootstrap_value

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns



@torch.no_grad()
def collect_rollout(
    env_adapter: GuardedTerritoryAdapter,
    policy: HetNetPolicy,
    critic: HetNetCritic,
    buffer: MAHACBuffer,
    n_scouts: int,
    n_interc: int,
):
    """collect T timesteps of exp across B envs"""

    buffer.reset()
    B = buffer.B
    T = buffer.T
    device = buffer.device

    # init lstm hidden states (zeros at start)
    hidden_s = policy.preprocess_scout.init_hidden(B, n_scouts, device)
    hidden_i = policy.preprocess_interc.init_hidden(B, n_interc, device)

    # get inital obs from env
    obs_s, state_s, obs_i, state_i, positions = env_adapter.hetnet_reset()

    for t in range(T):
        # build ssn input for ts
        ssn_input = build_ssn_input(
            n_scouts, n_interc,
            env_adapter.n_intruders,
            env_adapter.world_size,
            t/T,
            B, device,
        )

        # snapshot hidden states before forward pass
        # what ppo recomputation will use to reinit
        hidden_s_snap = {k: v.clone() for k,v in hidden_s.items()}
        hidden_i_snap = {k: v.clone() for k,v in hidden_i.items()}

        #policy forward pass
        scout_dist, interc_dist, h_ssn, hidden_s, hidden_i = policy(
            obs_s, state_s, obs_i, state_i, 
            positions, ssn_input, 
            hidden_s, hidden_i,
            n_scouts, n_interc
        )

        scout_actions, scout_log_probs = policy.sample_action(scout_dist)
        interc_actions, interc_log_probs = policy.sample_action(interc_dist)

        # critic values:
        values = critic(h_ssn)
        value_s = values['scout'].squeeze(-1)
        value_i = values['interc'].squeeze(-1)

        # store in buffer rewards and dones are place holders and filled after the step
        buffer.store(
            obs_s, state_s, scout_actions, scout_log_probs, value_s,
            obs_i, state_i, interc_actions, interc_log_probs, value_i,
            reward=torch.zeros(B, device=device),
            done=torch.zeros(B, device=device),
            positions=positions,
            ssn_input=ssn_input,
            hidden_s=hidden_s_snap,
            hidden_i=hidden_i_snap
        )

        #env step
        all_actions = combine_actions(scout_actions, interc_actions)
        reward, done, info = env_adapter.hetnet_step(all_actions)

        buffer.rewards[t] = reward
        buffer.dones[t] = done.float()

        # FIX-0: diagnostic — verify dones fire and reward scale
        if t < 3 or t == T - 1:
            print(
                f"  [DIAG] t={t} done_sum={done.float().sum().item():.0f}/{B} "
                f"reward_mean={reward.mean().item():.4f} "
                f"reward_min={reward.min().item():.4f} "
                f"reward_max={reward.max().item():.4f}",
                flush=True,
            )

        obs_s, state_s, obs_i, state_i, positions = env_adapter.get_obs()

        # reset LSTM hidden states for term envs
        if done.any():
            done_mask_s = done.unsqueeze(1).expand(-1, n_scouts).reshape(-1).bool()
            done_mask_i = done.unsqueeze(1).expand(-1, n_interc).reshape(-1).bool()

            for key in hidden_s:
                hidden_s[key][done_mask_s] = 0.0
            for key in hidden_i: 
                hidden_i[key][done_mask_i] = 0.0

    # FIX-0: rollout summary diagnostic
    total_dones = buffer.dones.sum().item()
    # FIX-P1-B5: use adapter's actual max_steps instead of hardcoded 100
    _ms = getattr(env_adapter, '_max_steps', 100)
    print(
        f"  [ROLLOUT SUMMARY] total_dones={total_dones:.0f} "
        f"expected_{_ms}step_eps≈{B * (T // _ms):.0f} "
        f"reward_global_mean={buffer.rewards.mean().item():.4f} "
        f"reward_global_std={buffer.rewards.std().item():.4f}",
        flush=True,
    )

    ssn_input_final = build_ssn_input(
        n_scouts, n_interc, 
        env_adapter.n_intruders,
        env_adapter.world_size,
        1.0, B, device
    )

    _, _, h_ssn_final, _, _ = policy(
        obs_s, state_s, obs_i, state_i, 
        positions, ssn_input_final,
        hidden_s, hidden_i,
        n_scouts, n_interc
    )

    bootstrap_values = critic(h_ssn_final)

    if critic.mode != 'centralized':
        return {
            'scout': bootstrap_values['scout'].squeeze(-1),
            'interc': bootstrap_values['interc'].squeeze(-1)
        }
        
    else:
        return bootstrap_values

