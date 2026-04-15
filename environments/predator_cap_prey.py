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
        self.interceptor_agent_fov = kwargs.get("interceptor_agent_fov", 0.8)
        self.interceptor_prey_fov = 0.0

        self.capture_radius = kwargs.get("capture_radius", 0.15)
        self.found_radius = kwargs.get("found_radius", 0.3)

        self.scout_speed = kwargs.get("scout_speed", 0.8)
        self.interceptor_speed = kwargs.get("interceptor_speed", 0.8)

        self.step_penalty = kwargs.get("step_penalty", -0.05)

        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=0.1,
            drag=0.25,
            dim_c=0,
            x_semidim=self.world_size,
            y_semidim=self.world_size,
        )

        self.scouts = []
        for i in range(self.n_scouts):
            agent = Agent(
                name=f"scout_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.075),
                max_speed=self.scout_speed,
                color=Color.BLUE,
                u_range=1.0,
            )
            agent.agent_type = SCOUT
            agent.type_id = 0
            world.add_agent(agent=agent)
            self.scouts.append(agent)

        self.interceptors = []
        for i in range(self.n_interceptors):
            agent = Agent(
                name=f"interceptor_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.09),
                max_speed=self.interceptor_speed,
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

        self._scout_found_prey = None 
        self._interc_capture_prey = None

        self._scout_type_oh = torch.tensor([1.0, 0.0], device=device)
        self._interc_type_oh = torch.tensor([0.0, 1.0], device=device)

        return world
    

    def reset_world_at(self, env_index: typing.Optional[int] = None):
        batch = self.world.batch_dim
        device = self.world.device

        if env_index is None:
            self._scout_found_prey = torch.zeros(
                batch, self.n_scouts, dtype=torch.bool, device=device
            )
            self._interc_capture_prey = torch.zeros(
                batch, self.n_interceptors, dtype=torch.bool, device=device
            )
        else:
            self._scout_found_prey[env_index] = False
            self._interc_capture_prey[env_index] = False

        # prey: random pos in inner 60% of the world
        inner = self.world_size * 0.6
        prey_pos = (torch.rand(
            (1,2) if env_index is not None else (batch, 2),
            device=device
        ) * 2 - 1) * inner

        if env_index is not None:
            self.prey.set_pos(prey_pos.squeeze(0), batch_index=env_index)
        else:
            self.prey.set_pos(prey_pos, batch_index=env_index)
        
        for agent in self.all_agents:
            angle = torch.rand(
                (1,1) if env_index is None else (batch, 1),
                device=device
            ) * 2 * torch.pi
            radius = self.world_size * (
                0.6 + 0.3 + torch.rand(
                    (1,1) if env_index is not None else (batch, 1),
                    device=device,
                )
            )

            pos = torch.cat([
                radius * torch.cos(angle),
                radius * torch.sin(angle),
            ], dim=-1)

            if env_index is not None:
                agent.set_pos(pos.squeeze(0), batch_index=env_index)
                agent.set_vel(torch.zeros(2, device=device), batch_index=env_index)
            else:
                agent.set_pos(pos, batch_index=env_index)
                agent.set_vel(torch.zeros(batch, 2, device=device))

    
    def observation(self, agent: Agent) -> torch.Tensor:
        """
        per-agent obs w class specific fov

        [vel(2) | pos(2) | type_oh(2) | rel_prey(2) | rel_others((N-1)*2)]
        """

        batch = self.world.batch_dim
        device = self.world.device

        pos = agent.state.pos

        if agent.agent_type == SCOUT:
            type_oh = self._scout_type_oh
            prey_fov = self.scout_fov
            agent_fov = self.scout_fov
        else:
            type_oh = self._interc_type_oh
            prey_fov = self.interceptor_prey_fov
            agent_fov = self.interceptor_agent_fov

        state = torch.cat([agent.state.vel, pos, type_oh], dim=-1)

        obs_parts = []

        prey_rel = self.prey.state.pos - pos
        prey_dist = torch.linalg.vector_norm(prey_rel, dim=-1, keepdim=True)
        prey_visible = (prey_dist <= prey_fov).float()  
        obs_parts.append(prey_rel * prey_visible)

        for other in self.all_agents:
            if other is agent:
                continue
            other_rel = other.state.pos - pos
            other_dist = torch.linalg.vector_norm(other_rel, dim=-1, keepdim=True)
            other_visible = (other_dist <= agent_fov).float()
            obs_parts.append(other_dist * other_visible)
        
        obs_portion = torch.cat(obs_parts, dim=-1)
        
        return torch.cat([state, obs_portion], dim=-1)
    
    def reward(self, agent: Agent) -> torch.Tensor:
        """
        shared team reward : computed once and cached
        """

        batch = self.world.batch_dim
        device = self.world.device

        if agent is self.all_agents[0]:

            prey_pos = self.prey.state.pos
            agent_dists = []
            for a in self.all_agents:
                d = torch.linalg.vector_norm(a.state.pos - prey_pos, dim=-1)
                agent_dists.append(d)
            agent_dists = torch.stack(agent_dists, dim=1)

            self._agent_at_prey = agent_dists < self.capture_radius

            rew = torch.full((batch), self.step_penalty, device=device)

            rew -= 0.1 * agent_dists.mean(dim=1)

            all_found = self._agent_at_prey.all(dim=1)
            rew += 5.0 * all_found.float()

            self._cached_reward = rew

        return self._cached_reward
    
    
        
    
        
