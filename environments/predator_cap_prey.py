"""
adapeted from the heterogenous PCP env from section 7 of the hetnet paper (Seraj et al. 2021)

original: 10x10 grid, predators (perception) + capture agents (action), 1 stationary prey
us: continuous 2D VMAS world, continuous force actions, still heterogenous
    we use scouts as the FOV limited observation class (perception / predator)
    we use the interceptos as the zero FOV for prey (action / capture)
    interceptors can see nearby objects / agents for collision avoidance

primary abalation env as GNN communication is necessary since interceptors have no info about prey location without scout communication

usage:
    from environments.predator_capture_prey import PredatorCapturePreyAdapter
    adapter = PredatorCapturePreyAdapter(num_envs=64, device="cuda")
"""

import typing 
import torch
import vmas 
from typing import Dict
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

SCOUT = "scout"
INTERCEPTOR = "interceptor"

class Scenario(BaseScenario):
    """
    predator-capture-prey: heterogenous cooperative search and capture
    """
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        # agent counts
        self.n_scouts = kwargs.get("n_scouts", 2)
        self.n_interceptors = kwargs.get("n_interceptors", 2)
        self.n_agents = self.n_scouts + self.n_interceptors

        self.world_size = kwargs.get("world_size", 10.0)

        #FOV asymmetry - core heterogeneity
        self.scout_fov = kwargs.get("scout_fov", 1.5)
        self.interceptor_fov = kwargs.get("interceptor_fov", 0.8)