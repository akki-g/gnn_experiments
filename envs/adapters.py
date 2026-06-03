"""
Future home of VMAS/PCP adapters.

Phase 1 only wraps the toy env.

VectorEnvAdapter provides the same reset/step interface as ToyMultiAgentEnv
but will eventually wrap VMAS vectorized environments for the pred-cap-prey
thesis experiments.  All methods are stubs until Step 2.
"""

from typing import Tuple, Dict, Any

import torch
from torch import Tensor


class VectorEnvAdapter:
    """
    Thin adapter that normalises any vectorised multi-agent environment
    to the (obs, state, reward, done, info) interface expected by the
    MAPPO training loop.

    Future subclasses: VMASAdapter, PredCapPreyAdapter.
    """

    def reset(self) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        obs   : [B, N, obs_dim]
        state : [B, state_dim]
        """
        raise NotImplementedError("Step 2")

    def step(
        self, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        Parameters
        ----------
        actions : [B, N] long.

        Returns
        -------
        obs    : [B, N, obs_dim]
        state  : [B, state_dim]
        reward : [B, N]
        done   : [B, N]
        info   : dict
        """
        raise NotImplementedError("Step 2")
