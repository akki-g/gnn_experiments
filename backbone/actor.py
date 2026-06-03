"""
Decentralized actor head.

Action space: DISCRETE (Categorical).  Each agent selects one of action_dim
discrete actions given its local embedding from the encoder + comm slot.

No global state is used by the actor — only the per-agent embedding h.
This enforces the CTDE (Centralised Training, Decentralised Execution) contract.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Actor(nn.Module):
    """
    Maps per-agent hidden embeddings to a Categorical action distribution.

    Shared weights are used across all agents (same Actor for all N agents).
    """

    def __init__(self, hidden_dim: int, action_dim: int, num_layers: int = 1):
        """
        Parameters
        ----------
        hidden_dim : input embedding dimension D (output of encoder + comm).
        action_dim : number of discrete actions.
        num_layers : number of MLP layers before the logit projection.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_layers = num_layers

        # Trunk: Linear(hidden_dim, hidden_dim) + Tanh, repeated num_layers times
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*trunk_layers)

        # Head with small init to keep initial policy near uniform
        self.head = nn.Linear(hidden_dim, action_dim)

        # Orthogonal init: gain=sqrt(2) on trunk, gain=0.01 on head
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, h: Tensor) -> torch.distributions.Categorical:
        """
        h [B, N, hidden_dim] → Categorical with logits [B, N, action_dim]

        The distribution is over the last dimension (action_dim).
        Each (b, n) entry is an independent Categorical.
        """
        assert h.shape[-1] == self.hidden_dim, (
            f"h last dim {h.shape[-1]} != hidden_dim {self.hidden_dim}"
        )
        logits = self.head(self.trunk(h))
        return torch.distributions.Categorical(logits=logits)

    def act(self, h: Tensor, deterministic: bool = False
            ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample or greedily select an action for each agent.

        Parameters
        ----------
        h             : [B, N, hidden_dim] embeddings.
        deterministic : if True, take argmax instead of sampling.

        Returns
        -------
        action   : [B, N] long  — selected discrete actions.
        log_prob : [B, N] float — log probability of selected actions.
        entropy  : [B, N] float — per-agent distribution entropy.
        """
        dist = self.forward(h)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, h: Tensor, actions: Tensor
                         ) -> Tuple[Tensor, Tensor]:
        """
        Evaluate log-probs and entropy for given actions under the current policy.

        Used during the PPO update to compute new_log_probs.
        NOTE: old_log_probs stored in the rollout buffer must NOT be passed here —
        they were already detached at collection time.

        Parameters
        ----------
        h       : [B, N, hidden_dim] or [(T*B), N, hidden_dim] embeddings.
        actions : [B, N] or [(T*B), N] long  — actions to evaluate.

        Returns
        -------
        log_probs : [B, N] or [(T*B), N] float.
        entropy   : [B, N] or [(T*B), N] float.
        """
        dist = self.forward(h)
        assert actions.shape == dist.logits.shape[:-1], (
            f"actions shape {actions.shape} != dist.logits.shape[:-1] {dist.logits.shape[:-1]}"
        )
        return dist.log_prob(actions), dist.entropy()
