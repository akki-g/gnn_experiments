"""
Shared MLP encoder.

Same encoder weights are used for all agents (parameter sharing by default).
The encoder maps raw per-agent observations to a fixed-size hidden embedding.
"""

import math

import torch.nn as nn
from torch import Tensor


class SharedMLPEncoder(nn.Module):
    """
    Shared MLP that encodes per-agent observations.

    Processes each agent's observation independently using the same weights.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, num_layers: int = 2,
                 activation: str = "tanh"):
        """
        Parameters
        ----------
        obs_dim    : dimension of each agent's local observation.
        hidden_dim : output embedding dimension D.
        num_layers : number of MLP layers (>= 1).
        activation : "tanh" or "relu".
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        act_cls = nn.Tanh if activation == "tanh" else nn.ReLU
        layers = [nn.Linear(obs_dim, hidden_dim), act_cls()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        self.net = nn.Sequential(*layers)

        # Orthogonal init with gain=sqrt(2), zero biases
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: Tensor) -> Tensor:
        """
        obs [B, N, obs_dim] or [T, B, N, obs_dim]
        → [B, N, hidden_dim] or [T, B, N, hidden_dim]
        """
        assert obs.shape[-1] == self.obs_dim, (
            f"obs last dim {obs.shape[-1]} != obs_dim {self.obs_dim}"
        )
        return self.net(obs)
