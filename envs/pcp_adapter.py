"""
PCPAdapter: VectorEnvAdapter subclass wrapping PCPEnv.

Thin subclass of PursuitAdapter that swaps PursuitEnv for PCPEnv and
forwards the two PCP-specific constructor parameters (detect_radius,
detection_window). All other behaviour — reset/step delegation, attribute
surface, agent_positions property, render stub — is inherited unchanged.

IMPORTANT: No VMAS import anywhere in this file.
"""

from typing import Tuple, Dict, Any

import torch
from torch import Tensor

from envs.pursuit_adapter import PursuitAdapter
from envs.pcp import PCPEnv


class PCPAdapter(PursuitAdapter):
    """
    Thin adapter that wraps PCPEnv to the MAPPO backbone VectorEnvAdapter interface.

    Inherits all delegation (reset, step, agent_positions, render) from PursuitAdapter.
    Only overrides __init__ to instantiate PCPEnv instead of PursuitEnv and accept
    the additional PCP parameters.

    Extra parameters (beyond PursuitAdapter)
    -----------------------------------------
    detect_radius    : float — detection radius for class-0 (detector) agents.
    detection_window : int  — number of steps the capture window stays open after detection.
    """

    def __init__(
        self,
        scenario: str = "pcp",           # ignored; kept for API parity
        num_envs: int = 32,
        n_agents: int = 5,
        max_steps: int = 100,
        seed: int = 0,
        device: str = "cpu",
        include_timestep: bool = True,
        # PCP-specific (required)
        detect_radius: float = 2.0,
        detection_window: int = 5,
        # Pursuit-compatible params forwarded to PCPEnv
        alpha: float = 0.0,
        num_full_sight: int = 3,
        observe_teammates: bool = False,
        world_size: float = 12.0,
        capture_radius: float = 0.6,
        move_step: float = 0.2,
        target_speed: float = 0.0,       # PCP forces 0.0 anyway; passed for clarity
        dist_coef: float = 0.1,
        step_penalty: float = 0.01,
        capture_bonus: float = 10.0,
        capture_k: int = 1,
    ):
        # Do NOT call super().__init__() — we build self.env ourselves to use PCPEnv.
        self.B = num_envs
        self.num_agents = n_agents
        self.device = torch.device(device)

        # Build the wrapped PCP environment
        self.env = PCPEnv(
            num_envs=num_envs,
            num_agents=n_agents,
            max_steps=max_steps,
            seed=seed,
            device=device,
            world_size=world_size,
            capture_radius=capture_radius,
            move_step=move_step,
            target_speed=target_speed,
            alpha=alpha,
            num_full_sight=num_full_sight,
            observe_teammates=observe_teammates,
            include_timestep=include_timestep,
            dist_coef=dist_coef,
            step_penalty=step_penalty,
            capture_bonus=capture_bonus,
            capture_k=capture_k,
            detect_radius=detect_radius,
            detection_window=detection_window,
        )

        # Mirror the attribute surface VMASAdapter exposes to the trainer
        self.obs_dim    = self.env.obs_dim
        self.action_dim = self.env.action_dim   # 5
        self.state_dim  = self.env.state_dim
        self.num_agents = n_agents
        self.N          = n_agents

        # Forward-compat: class_id [N] — comm slot will consume this.
        self.class_id = self.env.class_id
