import torch

class GNNRolloutBuffer:
    def __init__(self, gamma, gae_lambda, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.adj = []
        self.advantages = None
        self.returns = None
        self.device = device

    def add_timestep(self, obs, actions, rewards, dones, log_probs, values, A):
        self.obs.append(obs.to(self.device))
        self.actions.append(actions.to(self.device))
        self.rewards.append(rewards.to(self.device))
        self.dones.append(dones.to(self.device))
        self.log_probs.append(log_probs.to(self.device))
        self.values.append(values.to(self.device))
        self.adj.append(A.to(self.device))


    def compute_advantages(self, last_values):
        # Each stored tensor is (E, N, ...) - we flatten E*N for GAE
        buffer_size = len(self.obs)
        # Stack: (T, E, N, ...) for rewards, values, dones
        rewards = torch.stack(self.rewards).to(device=self.device, dtype=torch.float32)  # (T, E, N)
        values = torch.stack(self.values).to(device=self.device, dtype=torch.float32)    # (T, E, N)
        dones = torch.stack(self.dones).to(device=self.device, dtype=torch.float32)      # (T, E, N)

        T, E, N = rewards.shape
        # Flatten E and N together for GAE computation
        rewards_flat = rewards.reshape(T, E * N)
        values_flat = values.reshape(T, E * N)
        dones_flat = dones.reshape(T, E * N)
        last_values_flat = last_values.reshape(E * N).to(device=self.device, dtype=torch.float32)

        advantages = torch.zeros((T, E * N), dtype=torch.float32, device=self.device)
        last_gae = torch.zeros(E * N, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_values_flat
            else:
                next_value = values_flat[t + 1]
            deltas = rewards_flat[t] + self.gamma * (1 - dones_flat[t]) * next_value - values_flat[t]
            advantages[t] = deltas + self.gamma * self.gae_lambda * (1 - dones_flat[t]) * last_gae
            last_gae = advantages[t]

        returns = advantages + values_flat
        # Reshape back to (T, E, N)
        self.advantages = advantages.reshape(T, E, N)
        self.returns = returns.reshape(T, E, N)

    def get_batches(self, B, connectivity_bias=False):
        T = len(self.obs)
        E, N = self.obs[0].shape[0], self.obs[0].shape[1]

        obs = torch.stack(self.obs).to(device=self.device, dtype=torch.float32)       # (T, E, N, obs_dim)
        actions = torch.stack(self.actions).to(device=self.device, dtype=torch.float32) # (T, E, N, act_dim)
        log_probs = torch.stack(self.log_probs).to(device=self.device, dtype=torch.float32) # (T, E, N)
        values = torch.stack(self.values).to(device=self.device, dtype=torch.float32)   # (T, E, N)
        adj = torch.stack(self.adj).to(device=self.device, dtype=torch.float32)         # (T, E, N, N)

        # Merge T and E into one batch dimension: (T*E, N, ...)
        obs = obs.reshape(T * E, N, -1)
        actions = actions.reshape(T * E, N, -1)
        log_probs = log_probs.reshape(T * E, N)
        values = values.reshape(T * E, N)
        adj = adj.reshape(T * E, N, N)
        advantages = self.advantages.reshape(T * E, N)
        returns = self.returns.reshape(T * E, N)

        # Normalize advantages across entire buffer
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        if connectivity_bias:
            # adj is symmetric-normalized (no self-loops), so recover binary
            # connectivity and count actual edges — do NOT subtract N
            binary_adj = (adj.abs() > 1e-6).float()
            edge_counts = binary_adj.sum(dim=(-2, -1)) + 1.0  # Laplace smoothing
            edge_probs = edge_counts / edge_counts.sum()
            if torch.isfinite(edge_probs).all() and (edge_probs >= 0).all():
                perm = torch.multinomial(edge_probs, T * E, replacement=False)
            else:
                perm = torch.randperm(T * E, device=self.device)
        else:
            perm = torch.randperm(T * E, device=self.device)

        batches = perm.split(B)

        for idx in batches:
            yield (
                obs[idx], actions[idx], log_probs[idx],
                advantages[idx], returns[idx], adj[idx], values[idx],
            )

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.adj = []
        self.advantages = None
        self.returns = None