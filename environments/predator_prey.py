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
    
    def reset_world_at(self, env_index: typing.Optional[int]):
        batch = self.world.batch_dim
        device = self.world.device

        # reset tracking
        if env_index is None:
            self._agent_at_prey = torch.zeros(
                batch, self.n_agents, dtype=torch.bool, device=device
              )
        else:
            self._agent_at_prey[env_index] = False
        
        # prey- random positions in center 60% of the world
        inner = self.world_size * 0.6
        # FIX-P1-B8: scale prey_pos to [-inner, inner] (was [0, 1]; PCP already correct)
        prey_pos = (torch.rand(
            (1,2) if env_index is not None else (batch, 2),
            device=device
        ) * 2 - 1) * inner

        if env_index is not None:
            self.prey.set_pos(prey_pos.squeeze(0), batch_index=env_index)
        else:
            self.prey.set_pos(prey_pos, batch_index=env_index)

        #agents- random pos in 60-90% of the world
        for agent in self.all_agents:
            angle = torch.rand(
                (1,1) if env_index is None else (batch, 1),
                device=device
            ) * 2 * torch.pi

            radius = self.world_size * (
                0.6 + 0.3 * torch.rand(
                    (1,1) if env_index is None else (batch, 1),
                    device=device
                )
            )
            pos = torch.cat([
                radius * torch.cos(angle),
                radius * torch.sin(angle)
            ], dim=-1)

            if env_index is not None:
                agent.set_pos(pos.squeeze(0), batch_index=env_index)
                agent.set_vel(torch.zeros(2, device=device), batch_index=env_index)
            else:
                agent.set_pos(pos, batch_index=env_index)
                agent.set_vel(torch.zeros(batch, 2, device=device), batch_index=env_index)

        return
    
    def observation(self, agent: Agent) -> torch.Tensor:
        """
        Per-agent obs FOV-masked
        format: [vel(2), pos(2), type_oh(2), rel_prey(2), rel_others((N-1)*2)]

        from paper:
            each agents obs is a concatenated array of the state vectors
            of all grids within the agent's for
        us:
            relative positions of prey and other agents, zero-masked if outside the agents FOV
        """

        batch = self.world.batch_dim
        device = self.world.device

        fov = self.agent_fov # same fov for all 

        if agent.agent_type == SCOUT:
            type_oh = self._scout_type_oh.expand(batch, -1)
        else:
            type_oh = self._interc_type_oh.expand(batch, -1)

        pos = agent.state.pos

        state = torch.cat([agent.state.vel, pos, type_oh], dim=-1)

        obs_parts = []

        prey_rel = self.prey.state.pos - pos 
        prey_dist = torch.linalg.vector_norm(prey_rel, dim=-1, keepdim=True)
        prey_visible = (prey_dist <= fov).float()
        obs_parts.append(prey_rel*prey_visible)

        #other agents rel pos fov masked
        for other in self.all_agents:
            if other is agent:
                continue

            other_rel = other.state.pos - pos
            other_dist = torch.linalg.vector_norm(other_rel, dim=-1, keepdim=True)
            other_visible = (other_dist <= fov).float()
            obs_parts.append(other_rel * other_visible)

        obs_portion = torch.cat(obs_parts, dim=-1)

        return torch.cat([state, obs_portion], dim=-1)
    
    def reward(self, agent: Agent) -> torch.Tensor:
        """
        Shared team reward (computed once and cached)

        from paper:
            each predator agent receives a small penalty of -0.05 per timestep
            until the prey is found
        us:
            keep the -0.05 step penalty
            distance shaping: -mean(dist_to_prey) for all agents 
            completion bonus: +5.0 when all agents reach prey
        """
        batch = self.world.batch_dim
        device = self.world.device

        #compute once and cache
        if agent is self.all_agents[0]:

            # all agent dists to prey
            prey_pos = self.prey.state.pos
            agent_dists = []
            for a in self.all_agents:
                d = torch.linalg.vector_norm(a.state.pos - prey_pos, dim=-1)
                agent_dists.append(d)
            agent_dists = torch.stack(agent_dists, dim=1)

            self._agent_at_prey = agent_dists < self.capture_radius
            
            #step penalty
            rew = torch.full((batch,), self.step_penalty, device=device)

            #distance shaping: encourage all agents towards prey
            rew -= 0.1 * agent_dists.mean(dim=1)

            all_found = self._agent_at_prey.all(dim=1)
            rew += 5.0 * all_found.float()

            self._cached_reward = rew

        return self._cached_reward
    
    def done(self) -> torch.Tensor:
        """ep ends when all agents reached the prey"""

        if self._agent_at_prey is None:
            return torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            )

        return self._agent_at_prey.all(dim=1)
    
    def info(self, agent: Agent) -> dict:
        info = {}
        if self._agent_at_prey is not None:
            info["n_at_prey"] = self._agent_at_prey.sum(dim=-1).float()
            info["all_found"] = self._agent_at_prey.all(dim=1)

        return info
    

def get_obs_dim(n_scouts=2, n_intercepts=1):
    n_agents = n_scouts + n_intercepts

    return (
        2+2+2               # vel + pos + type_oh (state portion)
        +2                  # prey rel pos 
        +(n_agents - 1) * 2 # other agents rel pos
    )


# adapter 
class PredatorPreyAdapter:
    """
    wraps the predator prey vmas scenario w the same interface as the other envs
    meant to mirror petting zoo interface 
    
    Usage based on algo:
        HetNet:     hetnet_reset / hetnet_step / get_obs / build_hetero_adj
        GNN-MAPPO:  reset / step / build_adj / reset_env
        IPPO:       reset / step / reset_env
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        n_scouts: int = 2,
        n_interceptors: int = 1,
        max_steps: int = 80,
        world_size: float = 5.0,
        agent_fov: float = 1.5,
        capture_radius: float = 0.15,
        agent_speed: float = 0.8,
        step_penalty: float = -0.05,
        discrete: bool = False,
        n_actions: int = 5,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.n_scouts = n_scouts
        self.n_interceptors = n_interceptors
        self.n_defenders = n_scouts + n_interceptors
        self.discrete = discrete
        self._n_actions = n_actions

        self.n_intruders = 1
        self.n_zones = 1
        self._world_size = world_size

        if discrete:
            self.env = vmas.make_env(
                scenario=Scenario(),
                num_envs=num_envs,
                device=device,
                continuous_actions=False,
                max_steps=None,  # FIX-P1-B1: adapter owns all truncation via _step_counters; VMAS max_steps causes stale done=True after step 80
                n_scouts=n_scouts,
                n_interceptors=n_interceptors,
                world_size=world_size,
                agent_fov=agent_fov,
                capture_radius=capture_radius,
                agent_speed=agent_speed,
                step_penalty=step_penalty,
            )
        else:
            self.env = vmas.make_env(
                scenario=Scenario(),
                num_envs=num_envs,
                device=device,
                continuous_actions=True,
                clamp_actions=True,
                max_steps=None,  # FIX-P1-B1: adapter owns all truncation via _step_counters; VMAS max_steps causes stale done=True after step 80
                n_scouts=n_scouts,
                n_interceptors=n_interceptors,
                world_size=world_size,
                agent_fov=agent_fov,
                capture_radius=capture_radius,
                agent_speed=agent_speed,
                step_penalty=step_penalty,
            )

        self._obs_dim = get_obs_dim(n_scouts, n_interceptors)
        self._state_dim = 6   # vel(2) + pos(2) + type_oh(2)
        self._obs_portion_dim = self._obs_dim - self._state_dim
        self._action_dim = 2
        self._max_steps = max_steps

        self.defender_indices = list(range(self.n_defenders))

        self.agent_types = torch.tensor(
            [0] * n_scouts + [1] * n_interceptors,
            device=self.device,
        )

        self._cached_obs = None
        self._cached_pos = None
        # FIX-P1-B1: port step-counter FIX-1B from guided_coverage.py
        self._step_counters = torch.zeros(num_envs, device=self.device, dtype=torch.long)

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def obs_dim(self) -> int:
        return self._obs_dim
 
    @property
    def state_dim(self) -> int:
        return self._state_dim
 
    @property
    def obs_portion_dim(self) -> int:
        return self._obs_portion_dim
 
    @property
    def action_dim(self) -> int:
        return self._action_dim
 
    @property
    def n_agents(self) -> int:
        return self.n_defenders
 
    @property
    def world_size(self) -> float:
        return self._world_size
    
    # internal helpers
    def _stack_obs(self, all_obs):
        self._cached_obs = torch.stack(
            [all_obs[i] for i in self.defender_indices], dim=1
        )
        self._cached_pos = self._cached_obs[:, :, 2:4].clone()

    def _make_action_list(self, defender_actions: torch.Tensor):
        return [defender_actions[:, j, :] for j in range(self.n_defenders)]

    def _dispatch_actions(self, defender_actions: torch.Tensor):
        if self.discrete:
            return [defender_actions[:, j].long() for j in range(self.n_defenders)]
        else:
            return [defender_actions[:, j, :] for j in range(self.n_defenders)]
    
    # gnn-mappo / ippo interface
    def reset(self):
        all_obs = self.env.reset()
        self._step_counters.zero_()  # FIX-P1-B1: reset step counters on full reset
        self._stack_obs(all_obs)
        return self._cached_obs, self._cached_pos
    
    def step(self, defender_actions: torch.Tensor):
        """
        args: defender_actions (B, N, 2)
        returns: obs (B,N,obs_dim), rewards (B,N), dones (B,), info, positions (B, N, 2)
        """

        all_actions = self._dispatch_actions(defender_actions)
        all_obs, all_rew, dones, all_infos = self.env.step(all_actions)

        # FIX-P1-B1: port step-counter FIX-1B from guided_coverage.py
        self._step_counters += 1
        forced_dones = self._step_counters >= self._max_steps
        is_truncation = forced_dones & ~dones  # FIX-P1-B7: truncation flag for downstream GAE
        dones = dones | forced_dones
        self._step_counters[dones] = 0

        self._stack_obs(all_obs)

        rewards = torch.stack(
            [all_rew[i] for i in self.defender_indices], dim=1
        ) # (B,N)

        info = {}
        if len(all_infos) > 0:
            first_info = all_infos[0]
            if isinstance(first_info, dict):
                info = first_info

        # true done signaling: n_tagged = 1 when all agents at prey
        all_found = info.get("all_found", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        info["n_tagged"] = all_found.float()
        info["n_breached"] = torch.zeros(self.num_envs, device=self.device)
        # FIX-P1-B7: emit truncation flag for downstream GAE
        info["is_truncation"] = is_truncation

        return self._cached_obs, rewards, dones, info, self._cached_pos
    

    def reset_env(self):
        return self.reset()
    
    def build_adj(self, positions: torch.Tensor, r_comm: float) -> torch.Tensor:
        """
        Symmetric-normalized adjacency WITHOUT self-loops.
        S^0 = I in GraphConv handles self-features; self-loops here would double-count.
        Returns D^{-1/2} A D^{-1/2} where A has zero diagonal.
        positions: (num_envs, n_def, 2)
        """
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        adj = (dist <= r_comm).float()
        # Remove self-loops
        adj.diagonal(dim1=-2, dim2=-1).zero_()
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=-1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        return adj
    

    # hetnet interface
    def hetnet_reset(self):
        all_obs = self.env.reset()
        self._step_counters.zero_()  # FIX-P1-B1: reset step counters on full reset
        self._stack_obs(all_obs)
        return self.get_obs()
    
    def get_obs(self):
        """
        split cached obs into state (ego) and obs (other) portions
        """

        obs = self._cached_obs
        n_s = self.n_scouts
        sd = self.state_dim

        state_all = obs[:, :, :sd]
        obs_all = obs[:, :, sd:]

        state_s = state_all[:, :n_s, :]
        state_i = state_all[:, n_s:, :]
        obs_s = obs_all[:, :n_s, :]
        obs_i = obs_all[:, n_s:, :]

        return obs_s, state_s, obs_i, state_i, self._cached_pos
    
    def hetnet_step(self, defender_actions: torch.Tensor):
        """
        hetnet interface takes stacked actions and returns rewards, dones, info
        """

        all_actions = self._dispatch_actions(defender_actions)
        all_obs, all_rew, dones, all_info = self.env.step(all_actions)

        # FIX-P1-B1: port step-counter FIX-1B from guided_coverage.py
        self._step_counters += 1
        forced_dones = self._step_counters >= self._max_steps
        is_truncation = forced_dones & ~dones  # FIX-P1-B7: truncation flag for downstream GAE
        dones = dones | forced_dones
        self._step_counters[dones] = 0

        self._stack_obs(all_obs)

        reward = all_rew[0]

        info = {}
        if len(all_info) > 0:
            first_info = all_info[0]
            if isinstance(first_info, dict):
                info = first_info

        # FIX-P1-B7: emit truncation flag for downstream GAE
        info["is_truncation"] = is_truncation

        return reward, dones, info
    
    def build_hetero_adj(self, positions: torch.Tensor, r_comm: float) -> Dict[str, torch.Tensor]:
        """per edge type binary adj for hetgat layers"""

        n_s = self.n_scouts
        n_i = self.n_interceptors

        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        connected = (dist <= r_comm).float()

        #zero out self-loops
        eye = torch.eye(n_s + n_i, device=positions.device).unsqueeze(0)
        connected = connected * (1.0 - eye)

        adj = {
            'scout_to_scout': connected[:, :n_s, :n_s],
            'scout_to_inter': connected[:, :n_s, n_s:],
            'inter_to_scout': connected[:, n_s:, :n_s],    # rows=interc, cols=scout
            'inter_to_inter': connected[:, n_s:, n_s:],    # rows=interc, cols=interc
        }

        batch = positions.shape[0]
        adj['scout_to_ssn'] = torch.ones(batch, n_s, 1, device=self.device)
        adj['inter_to_ssn'] = torch.ones(batch, n_i, 1, device=self.device)

        return adj

