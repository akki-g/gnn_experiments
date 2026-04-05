"""
Predator-prey env from the original HetNet paper
For use of testing and baselining against ippo and gnn-mappo

Original: 10x10, N=3 predators, 1 stationary prey, discrete actions, FOV-limited
Ours: continuours 2d VMAS world, continuous force actions, still FOV-limited

Also changed agents to "scouts" and "interceptors" for better compatability
still have identical FOV and capabilities

usage:
    from environments.predator_prey import PredatorPreyAdapter
    adapter = PredatorPreyAdapter(num_env=..., device=..., n_scouts=..., n_interceptors=...)
"""

import typing
import torch
import vmas
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color 

from typing import Dict


# agent defs 
SCOUT = "scout"
INTERCEPTOR = "interceptor"


class Scenario(BaseScenario):
    """
    predator-prey: homogeneous cooperative search
    All agents (predators) must find and converge on a single stationary prey
    
    Paper Reference: Section 7 in Seraj at al. 2021
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        
        self.n_scouts = kwargs.get("n_scouts", 2)
        self.n_interceptors = kwargs.get("n_interceptors", 1)
        self.n_agents = self.n_scouts + self.n_interceptors

        self.world_size = kwargs.get("world_size", 5.0)

        # all agents have the same FOV
        self.agent_fov = kwargs.get("agent_fov", 1.5)

        self.capture_radius = kwargs.get("capture_radius", 0.15)

        self.agent_speed = kwargs.get("agent_speed", 0.8)

        # step penalty from paper 
        self.step_penalty = kwargs.get("step_penalty", -0.05)

        world = World(
            batch_dim=batch_dim,
            device=device,
            dt =0.1,
            drag=0.25,
            dim_c=0,
            x_semidim=self.world_size,
            y_semidim=self.world_size
        )

        self.scouts = []
        for i in range(self.n_scouts):
            agent = Agent(
                name=f"scout_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.075),
                max_speed=self.agent_speed,
                color=Color.BLUE,
                u_range=1.0
            )

            agent.agent_type = SCOUT
            agent.type_id = 1.0
            world.add_agent(agent)
            self.scouts.append(agent)

        self.interceptors = []
        for i in range(self.n_interceptors):
            agent = Agent(
                name=f"interceptor_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.075),
                max_speed=self.agent_speed,
                color=Color.GREEN,
                u_range=1.0,
            )
            agent.agent_type = INTERCEPTOR
            agent.type_id = 1
            world.add_agent(agent=agent)
            self.interceptors.append(agent)
 
        self.all_agents = self.scouts + self.interceptors

        self.prey = Landmark(
            name="prey_0",
            collide=False,
            movable=False,
            shape=Sphere(radius=0.12),
            color=Color.RED,
        )
        world.add_landmark(self.prey)
 
        # Tracking: which agents have reached the prey
        self._agent_at_prey = None
 
        # Cache type one-hots
        self._scout_type_oh = torch.tensor([1.0, 0.0], device=device)
        self._interc_type_oh = torch.tensor([0.0, 1.0], device=device)
 
        return world
