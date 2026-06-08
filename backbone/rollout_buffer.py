"""
On-policy rollout buffer for MAPPO.

Shape conventions (locked by thinker Q1)
-----------------------------------------
Symbol meanings:
    T         = rollout horizon (steps)
    B         = number of parallel environments (num_envs)
    N         = number of agents
    obs_dim   = per-agent local observation dimension
    state_dim = global state dimension (or N * obs_dim fallback)
    act_dim   = action dimension (unused for discrete; actions are [T,B,N] long)

Stored tensor shapes:
    obs          : [T, B, N, obs_dim]
    state        : [T, B, state_dim]
    actions      : [T, B, N]           long  (discrete action index)
    log_probs    : [T, B, N]           float (detached at collection time)
    values       : [T, B, N]           float (detached at collection time, per-agent Q3)
    rewards      : [T, B, N]           float (per-agent)
    dones        : [T, B, N]           float (1.0 = episode ended, 0.0 = ongoing)
    bad_masks    : [T, B, N]           float (0.0 = truncated time-limit end; 1.0 = normal)
    active_masks : [T, B, N]           float (0.0 = dead agent; 1.0 = alive)
    advantages   : [T, B, N]           float (computed by compute_returns_and_advantages)
    returns      : [T, B, N]           float (computed by compute_returns_and_advantages)
    comm_masks   : [T, B, N, N]        bool  (True = receiver i may receive from sender j;
                                              diagonal always False; default = full off-diagonal)

bad_masks / truncation semantics
----------------------------------
    bad_masks[t] = 0.0  episode ended at t due to time-limit (truncated):
                        the agent can still receive future rewards, so we
                        SHOULD bootstrap V(s_{t+1}) even though done[t]=1.
    bad_masks[t] = 1.0  episode ended truly (done=1) or did not end (done=0):
                        standard non-truncated termination; do not bootstrap.

GAE value-bootstrap coefficient at step t:
    bootstrap_coef[t] = 1 - dones[t] * bad_masks[t]
    - done=0, bad=1 → coef=1  (not done → always bootstrap)
    - done=1, bad=0 → coef=1  (truncated → still bootstrap)
    - done=1, bad=1 → coef=0  (truly terminal → no bootstrap)

GAE lambda-continuation coefficient uses 1-dones[t] (stop trace at any
episode boundary, whether truncated or truly terminal).

Note: correct truncation handling ideally requires the environment to provide
V(s_terminal) — the value of the actual final obs before reset — rather than
V(s_reset). For Phase 1 (toy env, no time-limit truncation) this does not
arise because bad_masks is always 1.0.

All stored tensors are detached (no gradient tracking).
The buffer is device-aware; all tensors live on the specified device.

Minibatch shape (get_minibatches output)
-----------------------------------------
Flatten T*B, keep N axis  (researcher decision 1):
    obs_mb          : [T*B, N, obs_dim]
    state_mb        : [T*B, state_dim]
    actions_mb      : [T*B, N]
    log_probs_mb    : [T*B, N]
    values_mb       : [T*B, N]
    advantages_mb   : [T*B, N]
    returns_mb      : [T*B, N]
    bad_masks_mb    : [T*B, N]
    active_masks_mb : [T*B, N]
    comm_mask_mb    : [T*B, N, N]
"""

from typing import Optional

import torch
from torch import Tensor


class RolloutBuffer:
    """
    Pre-allocated rollout buffer for on-policy MAPPO.

    All tensors are pre-initialised and live on `device`.
    Call insert() at each environment step (auto-increments the step pointer),
    then compute_returns_and_advantages() after the rollout is full, then
    get_minibatches() to yield mini-batches for the PPO update.
    """

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        discrete: bool = True,
    ):
        """
        Parameters
        ----------
        rollout_length : T — number of steps per rollout.
        num_envs       : B — parallel environments.
        num_agents     : N.
        obs_dim        : per-agent observation dimension.
        state_dim      : global state dimension.
        action_dim     : discrete: ignored (actions stored as long indices);
                         kept for API symmetry with future continuous support.
        device         : torch device string.
        discrete       : True (Phase 1 only supports discrete, Q2 decision).
        """
        self.T = rollout_length
        self.B = num_envs
        self.N = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.discrete = discrete

        # Step pointer — auto-incremented by insert(), reset by clear()
        self.step = 0

        self._allocate()

    def _allocate(self) -> None:
        """Pre-allocate all storage tensors on the target device."""
        T, B, N = self.T, self.B, self.N
        d = self.device

        self.obs = torch.zeros(T, B, N, self.obs_dim, device=d)
        self.state = torch.zeros(T, B, self.state_dim, device=d)
        self.actions = torch.zeros(T, B, N, device=d, dtype=torch.long)
        self.log_probs = torch.zeros(T, B, N, device=d)
        self.values = torch.zeros(T, B, N, device=d)
        self.rewards = torch.zeros(T, B, N, device=d)
        # dones: 0=not terminal, 1=episode ended (truncated or truly terminal)
        self.dones = torch.zeros(T, B, N, device=d)
        # bad_masks: 1=normal (default), 0=truncated time-limit episode end
        self.bad_masks = torch.ones(T, B, N, device=d)
        # active_masks: 1=alive (default), 0=dead agent
        self.active_masks = torch.ones(T, B, N, device=d)
        # comm_masks: True = receiver i may receive from sender j; diagonal=False.
        # Default is full off-diagonal communication (all-to-all minus self).
        self.comm_masks = torch.ones(T, B, N, N, dtype=torch.bool, device=d)
        self.comm_masks[:, :, torch.arange(N), torch.arange(N)] = False
        # Computed after rollout:
        self.advantages = torch.zeros(T, B, N, device=d)
        self.returns = torch.zeros(T, B, N, device=d)

    @property
    def is_full(self) -> bool:
        """True when the buffer has been filled for exactly T steps."""
        return self.step == self.T

    def insert(
        self,
        *,
        obs: Tensor,
        state: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        bad_mask: Optional[Tensor] = None,
        active_mask: Optional[Tensor] = None,
        comm_mask: Optional[Tensor] = None,
    ) -> None:
        """
        Store one step of experience at the current step pointer, then advance it.

        Shape assertions (all [B, N] or [B, state_dim]):
            obs        : [B, N, obs_dim]
            state      : [B, state_dim]
            action     : [B, N]          long
            log_prob   : [B, N]          float  — MUST be detached
            value      : [B, N]          float  — MUST be detached
            reward     : [B, N]          float
            done       : [B, N]          float  (1.0 = episode ended)
            bad_mask   : [B, N]          float  optional; 0=truncated, 1=normal
                         Defaults to all-ones (no truncation) if None.
            active_mask: [B, N]          float  optional; 0=dead, 1=alive
                         Defaults to all-ones (all alive) if None.
            comm_mask  : [B, N, N]       bool   optional; True = receiver i may
                         receive from sender j; diagonal must be False.
                         Defaults to full off-diagonal (all-to-all minus self) if None.
        """
        assert self.step < self.T, (
            f"Buffer is full (step={self.step} == T={self.T}). Call clear() before inserting."
        )
        assert obs.shape == (self.B, self.N, self.obs_dim), (
            f"obs shape {obs.shape} != ({self.B}, {self.N}, {self.obs_dim})"
        )
        assert state.shape == (self.B, self.state_dim), (
            f"state shape {state.shape} != ({self.B}, {self.state_dim})"
        )
        assert action.shape == (self.B, self.N), (
            f"action shape {action.shape} != ({self.B}, {self.N})"
        )
        assert log_prob.shape == (self.B, self.N), (
            f"log_prob shape {log_prob.shape} != ({self.B}, {self.N})"
        )
        assert value.shape == (self.B, self.N), (
            f"value shape {value.shape} != ({self.B}, {self.N})"
        )
        assert reward.shape == (self.B, self.N), (
            f"reward shape {reward.shape} != ({self.B}, {self.N})"
        )
        assert done.shape == (self.B, self.N), (
            f"done shape {done.shape} != ({self.B}, {self.N})"
        )
        assert not log_prob.requires_grad, "log_prob must be detached before inserting"
        assert not value.requires_grad, "value must be detached before inserting"

        t = self.step
        self.obs[t].copy_(obs)
        self.state[t].copy_(state)
        self.actions[t].copy_(action.long())
        self.log_probs[t].copy_(log_prob)
        self.values[t].copy_(value)
        self.rewards[t].copy_(reward)
        self.dones[t].copy_(done)
        if bad_mask is not None:
            assert bad_mask.shape == (self.B, self.N), (
                f"bad_mask shape {bad_mask.shape} != ({self.B}, {self.N})"
            )
            self.bad_masks[t].copy_(bad_mask)
        else:
            self.bad_masks[t].fill_(1.0)
        if active_mask is not None:
            assert active_mask.shape == (self.B, self.N), (
                f"active_mask shape {active_mask.shape} != ({self.B}, {self.N})"
            )
            self.active_masks[t].copy_(active_mask)
        else:
            self.active_masks[t].fill_(1.0)
        if comm_mask is not None:
            assert comm_mask.shape == (self.B, self.N, self.N), (
                f"comm_mask shape {comm_mask.shape} != ({self.B}, {self.N}, {self.N})"
            )
            self.comm_masks[t].copy_(comm_mask.bool())
        else:
            # Default: full off-diagonal (all-to-all minus self)
            self.comm_masks[t].fill_(True)
            self.comm_masks[t, :, torch.arange(self.N), torch.arange(self.N)] = False
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_value: Tensor,
        gamma: float,
        gae_lambda: float,
        value_normalizer=None,
    ) -> None:
        """
        Compute GAE advantages and discounted returns in-place.

        GAE with bad_masks for correct truncation handling:

            bootstrap_coef[t] = 1 - dones[t] * bad_masks[t]
                Ensures V(s_{t+1}) is still used when the episode was truncated
                (done=1, bad_mask=0) rather than truly terminated.

            delta_t  = reward_t + gamma * V_{t+1} * bootstrap_coef[t] - V_t
            adv_t    = delta_t + gamma * gae_lambda * (1 - dones[t]) * adv_{t+1}
            return_t = adv_t + V_t

        The GAE lambda trace stops at any episode boundary (1-dones[t]),
        regardless of whether the episode ended due to truncation or true termination.

        When value_normalizer is provided, self.values (stored normalized) are
        denormalized to raw reward space for GAE computation. self.values is NOT
        mutated; only a local denormalized copy is used. self.returns is in raw
        reward space when value_normalizer is provided (self.values stays normalized).

        Parameters
        ----------
        last_value       : [B, N]  — value estimate for the state AFTER the last step;
                           used to bootstrap non-terminal (or truncated) final states.
        gamma            : discount factor.
        gae_lambda       : GAE smoothing parameter.
        value_normalizer : optional ValueNorm instance; if provided, values and
                           last_value are denormalized before GAE computation so
                           that advantages and returns are in raw reward space.
                           If None (default), raw self.values are used directly.
        """
        assert self.is_full, (
            f"Buffer not full (step={self.step}, T={self.T}). "
            "Fill buffer completely before computing advantages."
        )
        assert last_value.shape == (self.B, self.N), (
            f"last_value shape {last_value.shape} != ({self.B}, {self.N})"
        )
        T = self.T
        if value_normalizer is None:
            values_for_gae = self.values
            last_value_for_gae = last_value
        else:
            values_for_gae = value_normalizer.denormalize(self.values)
            last_value_for_gae = value_normalizer.denormalize(last_value)
        gae = torch.zeros_like(last_value_for_gae)
        for t in reversed(range(T)):
            next_values = last_value_for_gae if t == T - 1 else values_for_gae[t + 1]
            bootstrap_coef = 1.0 - self.dones[t] * self.bad_masks[t]
            gae_cont = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * bootstrap_coef - values_for_gae[t]
            gae = delta + gamma * gae_lambda * gae_cont * gae
            self.advantages[t] = gae
        self.returns = self.advantages + values_for_gae

    def get_minibatches(self, num_minibatches: int):
        """
        Yield minibatches by randomly shuffling the T*B dimension.

        Flatten T*B keeping the N axis (researcher decision 1).
        The buffer must be fully filled and advantages must be computed
        before calling this method.

        Yields dicts with keys:
            obs, state, actions, log_probs, values, advantages, returns,
            bad_masks, active_masks, comm_mask
        Shapes: all [T*B // num_minibatches, N, ...] (or [T*B//M, ...] for state).
        """
        assert self.is_full, (
            f"Buffer not full (step={self.step}, T={self.T}). "
            "Cannot generate minibatches from a partially filled buffer."
        )
        total = self.T * self.B
        assert total % num_minibatches == 0, (
            f"total ({total}) must be divisible by num_minibatches ({num_minibatches})"
        )
        mb_size = total // num_minibatches

        obs_f = self.obs.reshape(total, self.N, self.obs_dim)
        state_f = self.state.reshape(total, self.state_dim)
        actions_f = self.actions.reshape(total, self.N)
        log_probs_f = self.log_probs.reshape(total, self.N)
        values_f = self.values.reshape(total, self.N)
        adv_f = self.advantages.reshape(total, self.N)
        ret_f = self.returns.reshape(total, self.N)
        bad_f = self.bad_masks.reshape(total, self.N)
        active_f = self.active_masks.reshape(total, self.N)
        comm_masks_f = self.comm_masks.reshape(total, self.N, self.N)

        perm = torch.randperm(total, device=self.device)
        for i in range(num_minibatches):
            idx = perm[i * mb_size:(i + 1) * mb_size]
            yield {
                "obs": obs_f[idx],
                "state": state_f[idx],
                "actions": actions_f[idx],
                "log_probs": log_probs_f[idx],
                "values": values_f[idx],
                "advantages": adv_f[idx],
                "returns": ret_f[idx],
                "bad_masks": bad_f[idx],
                "active_masks": active_f[idx],
                "comm_mask": comm_masks_f[idx],
            }

    def clear(self) -> None:
        """
        Reset the step pointer and reinitialise all tensors for the next rollout.

        bad_masks and active_masks are reset to all-ones (default safe values).
        comm_masks is reset to full off-diagonal (all-to-all minus self).
        """
        self.obs.zero_()
        self.state.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.bad_masks.fill_(1.0)
        self.active_masks.fill_(1.0)
        # Reset comm_masks to full off-diagonal communication
        self.comm_masks.fill_(True)
        self.comm_masks[:, :, torch.arange(self.N), torch.arange(self.N)] = False
        self.advantages.zero_()
        self.returns.zero_()
        self.step = 0
