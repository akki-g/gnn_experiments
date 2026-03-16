import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class IPPOPolicy(nn.Module):
    """Shared-weight MLP policy baseline without graph communication."""

    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.fc5.weight, gain=0.01)
        nn.init.constant_(self.fc5.bias, 0.0)

    def forward(self, obs):
        device = next(self.parameters()).device
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.to(device=device, dtype=torch.float32)

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.relu(self.fc4(x))
        return self.fc5(x)

    def get_actions(self, obs):
        mean = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions):
        mean = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        raw_actions = torch.atanh(actions.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
