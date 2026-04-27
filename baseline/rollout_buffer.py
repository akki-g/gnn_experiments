import torch


class IPPORolloutBuffer:
    """
    Multi-environment rollout buffer for IPPO.

    Stores tensors of shape (E, N, ...) per timestep, where E = num_envs
    and N = n_defenders. Mirrors GNNRolloutBuffer exactly, minus the
    adjacency matrix field.
    """

    def __init__(self, gamma, gae_lambda, device, discrete: bool = False):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.discrete = discrete
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None

    def add_timestep(self, obs, actions, rewards, dones, log_probs, values):
        """All inputs expected as (E, N, ...) tensors already on the correct device."""
        self.obs.append(obs.to(self.device))
        self.actions.append(actions.to(self.device))
        self.rewards.append(rewards.to(self.device))
        self.dones.append(dones.to(self.device))
        self.log_probs.append(log_probs.to(self.device))
        self.values.append(values.to(self.device))

    def compute_advantages(self, last_values):
        """
        GAE computation over the full (T, E, N) buffer.
        last_values: (E, N) critic estimates for the state after the last step.
        """
        rewards = torch.stack(self.rewards).to(dtype=torch.float32)  # (T, E, N)
        values  = torch.stack(self.values).to(dtype=torch.float32)   # (T, E, N)
        dones   = torch.stack(self.dones).to(dtype=torch.float32)    # (T, E, N)

        T, E, N = rewards.shape

        # Flatten E and N together for the GAE recurrence
        rewards_flat    = rewards.reshape(T, E * N)
        values_flat     = values.reshape(T, E * N)
        dones_flat      = dones.reshape(T, E * N)
        last_values_flat = last_values.reshape(E * N).to(device=self.device, dtype=torch.float32)

        advantages = torch.zeros((T, E * N), dtype=torch.float32, device=self.device)
        last_gae   = torch.zeros(E * N,       dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            next_value = last_values_flat if t == T - 1 else values_flat[t + 1]
            deltas = (
                rewards_flat[t]
                + self.gamma * (1 - dones_flat[t]) * next_value
                - values_flat[t]
            )
            advantages[t] = deltas + self.gamma * self.gae_lambda * (1 - dones_flat[t]) * last_gae
            last_gae = advantages[t]

        returns = advantages + values_flat

        # Reshape back to (T, E, N)
        self.advantages = advantages.reshape(T, E, N)
        self.returns    = returns.reshape(T, E, N)

    def get_batches(self, B):
        """
        Yield mini-batches of size B, shuffled over the T*E dimension.
        Each batch yields: obs, actions, log_probs, advantages, returns, old_values
        with shapes (B, N, ...).
        """
        T = len(self.obs)
        E = self.obs[0].shape[0]
        N = self.obs[0].shape[1]

        obs       = torch.stack(self.obs).to(device=self.device, dtype=torch.float32)       # (T, E, N, obs_dim)
        log_probs = torch.stack(self.log_probs).to(device=self.device, dtype=torch.float32) # (T, E, N)
        values    = torch.stack(self.values).to(device=self.device, dtype=torch.float32)    # (T, E, N)

        if self.discrete:
            actions = torch.stack(self.actions).to(device=self.device, dtype=torch.long)    # (T, E, N)
        else:
            actions = torch.stack(self.actions).to(device=self.device, dtype=torch.float32) # (T, E, N, act_dim)

        # Merge T and E into one shuffleable batch dimension
        obs       = obs.reshape(T * E, N, -1)
        if self.discrete:
            actions = actions.reshape(T * E, N)      # (T*E, N) — no trailing dim
        else:
            actions = actions.reshape(T * E, N, -1)  # (T*E, N, act_dim)
        log_probs = log_probs.reshape(T * E, N)
        values    = values.reshape(T * E, N)
        advantages = self.advantages.reshape(T * E, N)
        returns    = self.returns.reshape(T * E, N)

        # Normalize advantages across the whole buffer
        adv_mean = advantages.mean()
        adv_std  = advantages.std().clamp(min=1e-4)  # FIX-P3-B10: floor prevents amplification when advantages near-zero
        advantages = (advantages - adv_mean) / adv_std

        perm    = torch.randperm(T * E, device=self.device)
        batches = perm.split(B)

        for idx in batches:
            yield (
                obs[idx],
                actions[idx],
                log_probs[idx],
                advantages[idx],
                returns[idx],
                values[idx],
            )

    def clear(self):
        self.obs        = []
        self.actions    = []
        self.rewards    = []
        self.dones      = []
        self.log_probs  = []
        self.values     = []
        self.advantages = None
        self.returns    = None