"""
Centralised critic for MAPPO.

IMPORTANT (Q5 isolation rule):
    The critic input is RAW global state (or the concat-all-obs fallback).
    It does NOT receive comm-slot output.
    This is a hard architectural invariant: swapping comm modules must not
    change what the critic sees, ensuring fair ablations.

Critic output shape: [B, num_agents]  (per-agent values, Q3 decision).
"""

import math

import torch.nn as nn
from torch import Tensor


class CentralCritic(nn.Module):
    """
    Centralised value network.

    Inputs : raw global state [B, state_dim] or concat-obs fallback
             [B, N * obs_dim]  — see MAPPO.build_state_from_obs().
    Outputs: per-agent values [B, num_agents].

    The critic is NOT conditioned on comm-slot output.  This is the Q5
    isolation rule: the comm module only affects the actor path.
    """

    def __init__(self, state_dim: int, hidden_dim: int, num_agents: int,
                 num_layers: int = 2):
        """
        Parameters
        ----------
        state_dim  : dimension of the global state / concat-obs fallback.
        hidden_dim : MLP hidden width.
        num_agents : N — number of agents, equals output width.
        num_layers : depth of the MLP.
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.num_layers = num_layers

        # Build MLP: Linear(state_dim, hidden_dim) + Tanh
        #            + (Linear(hidden_dim, hidden_dim) + Tanh) * (num_layers - 1)
        #            + Linear(hidden_dim, num_agents)
        layers = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, num_agents))
        self.net = nn.Sequential(*layers)

        # Orthogonal init: gain=sqrt(2) on hidden layers, gain=1.0 on output head
        linear_count = 0
        total_linears = num_layers + 1  # num_layers hidden + 1 output
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                linear_count += 1
                if linear_count == total_linears:
                    # Output head: gain=1.0
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:
                    # Hidden layers: gain=sqrt(2)
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: Tensor) -> Tensor:
        """
        Parameters
        ----------
        state : [B, state_dim]  — RAW global state, NOT comm-modulated.

        Returns
        -------
        values : [B, num_agents]  per-agent value estimates.
        """
        assert state.shape[-1] == self.state_dim, (
            f"state last dim {state.shape[-1]} != state_dim {self.state_dim}"
        )
        return self.net(state)
