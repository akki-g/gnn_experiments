import torch


class IPPORolloutBuffer:
    def __init__(self, gamma, gae_lambda, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None

    def add_timestep(self, obs, actions, rewards, dones, log_probs, values):
        self.obs.append(obs.to(self.device))
        self.actions.append(actions.to(self.device))
        self.rewards.append(rewards.to(self.device))
        self.dones.append(dones.to(self.device))
        self.log_probs.append(log_probs.to(self.device))
        self.values.append(values.to(self.device))

    def compute_advantages(self, last_values):
        n_agents = self.obs[0].shape[0]
        buffer_size = len(self.obs)
        rewards = torch.stack(self.rewards).to(device=self.device, dtype=torch.float32)
        values = torch.stack(self.values).to(device=self.device, dtype=torch.float32)
        dones = torch.stack(self.dones).to(device=self.device, dtype=torch.float32)

        advantages = torch.zeros((buffer_size, n_agents), dtype=torch.float32, device=self.device)
        last_gae = torch.zeros(n_agents, dtype=torch.float32, device=self.device)

        for t in reversed(range(buffer_size)):
            if t == buffer_size - 1:
                if torch.is_tensor(last_values):
                    next_value = last_values.to(device=self.device, dtype=torch.float32)
                else:
                    next_value = torch.as_tensor(last_values, dtype=torch.float32, device=self.device)
            else:
                next_value = values[t + 1]
            deltas = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
            advantages[t] = deltas + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            last_gae = advantages[t]

        self.returns = advantages + values
        self.advantages = advantages

    def get_batches(self, batch_size):
        perm = torch.randperm(len(self.obs), device=self.device)
        batches = perm.split(batch_size)

        obs = torch.stack(self.obs).to(device=self.device, dtype=torch.float32)
        actions = torch.stack(self.actions).to(device=self.device, dtype=torch.float32)
        log_probs = torch.stack(self.log_probs).to(device=self.device, dtype=torch.float32)

        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        advantages = (self.advantages - adv_mean) / adv_std

        for idx in batches:
            yield (
                obs[idx],
                actions[idx],
                log_probs[idx],
                advantages[idx],
                self.returns[idx],
            )

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None
