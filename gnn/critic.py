import torch
import torch.nn.functional as F
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, device=None):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.device = device
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc2.weight, gain=2**0.5)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device, dtype=torch.float32)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
