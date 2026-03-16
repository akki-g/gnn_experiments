import torch
import torch.nn.functional as F
import torch.nn as nn

class ActionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=2**0.5)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, G):
        G = F.relu(self.fc1(G))
        G = self.fc2(G)
        return G