"""
Modified simple spread type of env 
Similar to guarded_territory (just with no intruders):
2 classes of agents:
    - Scouts: Larger FOV 
    - Interceptor: Smaller FOV (primary goal is to reach the landmarks)

Landmarks:
    - Randomly scattered through out the center of the world
    - More landmarks than scouts, so interceptors must help with coverage

Key Assymetry: Only interceptors count for coverage reward. This forces our GNN architectures to be the only source of communication

Reward Structure: 
    team_reward = -mean(min_interceptor_distance_to_each_landmark)
"""

from typing import Dict
import torch
import vmas
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

# agent type definitions
SCOUT = "scout"
INTERCEPTOR = "interceptor"


class Scenario(BaseScenario):
    """
    simple spread but harder
    """
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # agent / world config
        self.n_scouts = kwargs.get("n_scouts", 2)
        self.n_intercs = kwargs.get("n_intercs", 2)
        self.n_zones = kwargs.get("n_zones", 3)
        self.world_size = kwargs.get("world_size", 5.0)

        # obs radii 
        self.scout_fov = kwargs.get("scout_fov", 1.0)
        self.interc_fov = kwargs.get("interc_fov", 0.5)
        self.tag_radius = kwargs.get("tag_radius", 0.1)

        # speeds
        self.scout_speed = kwargs.get("scout_speed", 0.65)
        self.interc_speed = kwargs.get("interc_speed", 0.8)

        self.state_obs = kwargs.get("state_obs", False)

        world = World(
            batch_dim=batch_dim,
            device=device, 
            dt=0.1,
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
                max_speed=self.scout_speed,
                color=Color.BLUE,
                u_range=1.0,
            )
            agent.agent_type = SCOUT
            agent.type_id = 0
            world.add_agent(agent=agent)
            self.scouts.append(agent)
        
        self.intercs = []
        for i in range(self.n_intercs):
            agent = Agent(
                name=f"interc_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.09),
                max_speed=self.interc_speed,
                color=Color.GREEN,
                u_range=1.0,
            )
            agent.agent_type = INTERCEPTOR
            agent.type_id = 1
            world.add_agent(agent)
            self.intercs.append(agent)

        self.zones = []
        for i in range(self.n_zones):
            zone = Landmark(
                name=f"zone_{i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=0.2),
                color=Color.LIGHT_GREEN,
            )
            world.add_landmark(zone)
            self.zones.append(zone)

        
        self._scout_type_oh = torch.tensor([1.0, 0.0], device=device).unsqueeze(0)
        self._interc_type_oh = torch.tensor([0.0, 1.0], device=device).unsqueeze(0)

        return world

    def reset_world_at(self, env_index = None): 
        """
        Place landmarks in inner region, agents near the border
        """
        batch = self.world.batch_dim if env_index is None else 1
        device = self.world.device

        # landmarks 
        inner = self.world_size * 0.6
        for zone in self.zones:
            pos = (torch.rand(batch, 2, device=device) * 2 - 1) * inner
            if env_index is not None:
                zone.set_pos(pos.squeeze(0), batch_index=env_index)
            else:
                zone.set_pos(pos, batch_index=env_index)

        #scouts: random in outer ring
        for scout in self.scouts:
            angle = torch.rand(batch, 1, device=device) * 2 * torch.pi
            radius = self.world_size * (0.6 + 0.3 * torch.rand(batch, 1, device=device))
            pos = torch.cat([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
            if env_index is not None:
                scout.set_pos(pos.squeeze(0), batch_index=env_index)
            else:
                scout.set_pos(pos, batch_index=env_index)
            scout.set_vel(torch.zeros(batch, 2, device=device).squeeze(0) if env_index is not None 
                         else torch.zeros(batch, 2, device=device), 
                         batch_index=env_index if env_index is not None else None)

        # interc: random in outer ring but diff angle
        for interc in self.intercs:
            angle = torch.rand(batch, 1, device=device) * 2 * torch.pi
            radius = self.world_size * (0.6 + 0.3 * torch.rand(batch, 1, device=device))
            pos = torch.cat([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
            if env_index is not None:
                interc.set_pos(pos.squeeze(0), batch_index=env_index)
            else:
                interc.set_pos(pos, batch_index=env_index)
            interc.set_vel(torch.zeros(batch, 2, device=device).squeeze(0) if env_index is not None 
                          else torch.zeros(batch, 2, device=device),
                          batch_index=env_index if env_index is not None else None)


    def observation(self, agent: Agent):
        """
        returns obs for one agent, shape: (batch, obs_dim)

        format: [vel(2) | pos(2) | type_oh(2) | rel_landmarks(n_zones*2) | rel_defenders(n_def-1)*2]
        Entities outside the agent's FOV are zero-masked
        """
        batch = self.world.batch_dim
        device = self.world.device
        
        if agent.agent_type == SCOUT:
            fov = self.scout_fov
        else:
            fov = self.interc_fov

        # state portion
        vel = agent.state.vel
        pos = agent.state.pos

        if agent.agent_type == SCOUT:
            type_oh = self._scout_type_oh.expand(batch, -1)
        else:
            type_oh = self._interc_type_oh.expand(batch, -1)

        state = torch.cat([vel, pos, type_oh], dim=-1)

        # obs portion: fov masked
        landmark_obs = []
        for zone in self.zones:
            rel = zone.state.pos - pos
            dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
            in_fov = (dist <= fov).float()
            landmark_obs.append(rel*in_fov)

        # other defenders 
        all_defenders = self.scouts + self.intercs
        defender_obs = []
        for other in all_defenders:
            if other is agent:
                continue
            rel = other.state.pos - pos
            dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
            in_fov = (dist <= fov).float()
            defender_obs.append(rel*in_fov)

        obs_portion = torch.cat(landmark_obs + defender_obs, dim=-1)

        return torch.cat([state, obs_portion], dim=-1)
    
    def reward(self, agent: Agent):
        """
        shared team reward: negative mean of min interceptor distance to each landmark.
        plus a small collision penalty
        both classes receives the same rewards
        """

        batch = self.world.batch_dim
        device = self.world.device

        # only compute once per step, scouts[0] is always called first
        if agent is self.scouts[0]:
 
            interc_pos = torch.stack([a.state.pos for a in self.intercs], dim=1)
            landmark_pos = torch.stack([l.state.pos for l in self.zones], dim=1)
 
            diff = landmark_pos.unsqueeze(2) - interc_pos.unsqueeze(1)
            dists = torch.linalg.vector_norm(diff, dim=-1)
 
            min_dists, _ = dists.min(dim=2)
 
            coverage_reward = -min_dists.mean(dim=1)
 
            all_defenders = self.scouts + self.intercs
            collision_penalty = torch.zeros(batch, device=device)
            for i, a1 in enumerate(all_defenders):
                for a2 in all_defenders[i+1:]:
                    d = torch.linalg.vector_norm(
                        a1.state.pos - a2.state.pos, dim=-1
                    )
                    collision_penalty += (d < (a1.shape.radius + a2.shape.radius)).float() * 0.1
            
            self._cached_reward = coverage_reward - collision_penalty
 
        return self._cached_reward
    def done(self):
        """
        no early term episodes run for max_steps
        """
        return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)
    
    def info(self, agent: Agent):
        """return diagnostic info (only meaningful for the first agent)"""
        if agent is self.scouts[0]:
            interc_pos = torch.stack([a.state.pos for a in self.intercs], dim=1)
            zone_pos = torch.stack([l.state.pos for l in self.zones], dim=1)
            diff = zone_pos.unsqueeze(2) - interc_pos.unsqueeze(1)
            dists = torch.linalg.vector_norm(diff, dim=-1)
            min_dists, _ = dists.min(dim=2)

            return {
                "mean_landmark_dist": min_dists.mean(dim=1),
                "n_covered": (min_dists < 0.3).float().sum(dim=1),
            }
        
        return {}
    

class GuidedCoverageAdapter:
    """
    Wraps the guided coverage VMAS scenario similar to pettingZoo's interface
    cuz thats how i wrote the original baselines
    """

    def __init__(
            self,
            num_envs: int,
            device: torch.device,
            n_scouts: int = 2,
            n_intercs: int = 2,
            n_zones: int = 2,
            max_steps: int = 100,
            world_size: float = 2.0,
            scout_fov: float = 1.5,
            interc_fov: float = 0.3,
            discrete: bool = False,
            n_actions: int = 5,
    ):
        self.device = device
        self.n_scouts = n_scouts
        self.n_interceptors = n_intercs
        self.n_intruders = 0 # allows it to be compatible w the build_ssn_input method
        self.n_zones = n_zones
        self._world_size = world_size
        self.discrete = discrete
        self._n_actions = n_actions

        self.n_defenders = n_scouts + n_intercs

        if discrete:
            self.env = vmas.make_env(
                scenario=Scenario(),
                num_envs=num_envs,
                device=device,
                continuous_actions=False,
                max_steps=None,  # FIX-P1-B1: adapter owns all truncation via _step_counters; VMAS never resets self.steps after done, causing stale done signals
                n_scouts=n_scouts,
                n_intercs=n_intercs,
                n_zones=n_zones,
                world_size=world_size,
                scout_fov=scout_fov,
                interc_fov=interc_fov
            )
        else:
            self.env = vmas.make_env(
                scenario=Scenario(),
                num_envs=num_envs,
                device=device,
                continuous_actions=True,
                clamp_actions=True,
                max_steps=None,  # FIX-P1-B1: adapter owns all truncation via _step_counters; VMAS never resets self.steps after done, causing stale done signals
                n_scouts=n_scouts,
                n_intercs=n_intercs,
                n_zones=n_zones,
                world_size=world_size,
                scout_fov=scout_fov,
                interc_fov=interc_fov
            )

        # FIX-1B: explicit step counter so dones fire reliably at max_steps
        # (guards against VMAS version differences or scenario.done() always returning False)
        self._max_steps = max_steps
        self._step_counters = torch.zeros(num_envs, device=device, dtype=torch.long)

        self.num_envs = num_envs

        # all agents are defenders
        self.defender_indices = list(range(self.n_defenders))
        self._n_agents_total = len(self.env.agents)
        self._action_dim = 2

        self.agents_types = torch.tensor(
            [0]*n_scouts + [1] * n_intercs,
            device=device
        )

        # obs dim from test obs
        test_obs = self.env.reset()
        sample = test_obs[0]
        self._obs_dim = sample.shape[-1]
        self._state_dim = 6
        self._obs_portion_dim = self._obs_dim - self._state_dim

        # internal caches 
        self._cached_obs = None
        self._cached_pos = None

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
    def world_size(self) -> float:
        return self._world_size
    
    @property
    def n_agents(self) -> int:
        return self.n_defenders
    

    # internal helpers
    def _stack_defender_obs(self, all_obs):
        """ stack per agent obs list into (B, n_def, obs_dim) tensor and cache"""
        defender_obs = torch.stack(
            [all_obs[i] for i in self.defender_indices], dim=1
        )
        self._cached_obs = defender_obs
        self._cached_pos = defender_obs[:, :, 2:4].clone()
        return defender_obs
    
    def _make_action_list(self, defender_actions: torch.Tensor):
        """
        convert (B, n_def, 2) tensor into list of (B, 2) tensors
        no intruder padding needed, all agents are defenders
        """
        return [defender_actions[:, i] for i in range(self.n_defenders)]

    def _dispatch_actions(self, defender_actions: torch.Tensor):
        if self.discrete:
            return [defender_actions[:, j].long() for j in range(self.n_defenders)]
        else:
            return [defender_actions[:, j] for j in range(self.n_defenders)]

    # gnn-mappo and ippo interface

    def reset(self):
        """
        returns: obs (B, n_def, obs_dim), pos (B, n_def, 2)
        """
        all_obs = self.env.reset()
        defender_obs = self._stack_defender_obs(all_obs)
        self._step_counters.zero_()  # FIX-1B: reset step counters on full reset
        return defender_obs, self._cached_pos
    
    def reset_env(self):
        """
        alias for reset env for gnn-mappo and ippo
        """
        return self.reset()

    def step(self, defender_actions: torch.Tensor):
        """
        gnn-mappo / ippo interface
        args: defender_actions (B, n_def, 2)
        returns: obs, rewards, dones, info, positions
        """

        all_actions = self._dispatch_actions(defender_actions)
        all_obs, all_rew, dones, all_infos = self.env.step(all_actions)

        # FIX-1B: force dones at max_steps regardless of VMAS internal behavior
        self._step_counters += 1
        forced_dones = self._step_counters >= self._max_steps
        dones = dones | forced_dones
        self._step_counters[dones] = 0  # reset counters for done envs (VMAS auto-resets them)

        defender_obs = self._stack_defender_obs(all_obs)

        # per-agent rewards: broadcast shared reward to (B, n_def)

        defender_rewards = torch.stack(
            [all_rew[i] for i in self.defender_indices], dim=1
        )

        # info needs to include n_tagged/n_breached for minimal code changes elsewhere
        info = {}
        if len(all_infos) > 0:
            first_info = all_infos[0]
            if isinstance(first_info, dict):
                info = first_info

        if "n_tagged" not in info:
            info["n_tagged"] = torch.zeros(self.num_envs, device=self.device)
        if "n_breached" not in info:
            info["n_breached"] = torch.zeros(self.num_envs, device=self.device)
        # FIX-P1-B7: all GC dones are truncations (scenario.done() always returns False)
        info["is_truncation"] = forced_dones

        return defender_obs, defender_rewards, dones, info, self._cached_pos
    
    def build_adj(self, position: torch.Tensor, r_comm: float) -> torch.Tensor:
        """
        Symmetric-normalized adjacency WITHOUT self-loops.
        S^0 = I in GraphConv handles self-features; self-loops here would double-count.
        Returns D^{-1/2} A D^{-1/2} where A has zero diagonal.
        positions: (num_envs, n_def, 2)
        """
        diff = position.unsqueeze(2) - position.unsqueeze(1)
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
    
    # HetNet interface
    def hetnet_reset(self):
        """
        reset env, cache obs, return split obs
        returns: (obs_s, state_s, obs_i, state_i, positions)
        """

        all_obs = self.env.reset()
        self._stack_defender_obs(all_obs)
        self._step_counters.zero_()  # FIX-1B: reset step counters
        return self.get_obs()
    
    def get_obs(self):
        """
        split cached obs by agent class into separate tensors
        returns: (obs_s, state_s, obs_i, state_i, positions)
        """
        obs = self._cached_obs
        n_s = self.n_scouts
        sd = self._state_dim

        state_all = obs[:, :, :sd]
        obs_all = obs[:, :, sd:]

        state_s = state_all[:, :n_s, :]
        state_i = state_all[:, n_s:, :]
        obs_s = obs_all[:, :n_s, :]
        obs_i = obs_all[:, n_s:, :]

        return obs_s, state_s, obs_i, state_i, self._cached_pos
    
    def hetnet_step(self, defender_actions: torch.Tensor):
        """
        hetnet interface: takes stacked actoins, returns (reward, done, info)
        args: defender_actions (B, n_def, 2) - scouts first then interceptors
        returns: 
            reward: (B, ) - shared team reward
            done: (B, ) bool
            info: dict 
        """
        
        all_actions = self._dispatch_actions(defender_actions)
        all_obs, all_rew, dones, all_infos = self.env.step(all_actions)

        # FIX-1B: force dones at max_steps regardless of VMAS internal behavior
        self._step_counters += 1
        forced_dones = self._step_counters >= self._max_steps
        dones = dones | forced_dones
        self._step_counters[dones] = 0  # reset counters for done envs (VMAS auto-resets them)

        self._stack_defender_obs(all_obs)

        # shared team reward from cache
        reward = all_rew[0]

        info = {}
        if len(all_infos) > 0:
            first_info = all_infos[0]
            if isinstance(first_info, dict):
                info = first_info

        # FIX-P1-B7: all GC dones are truncations (scenario.done() always returns False)
        info["is_truncation"] = forced_dones

        return reward, dones, info
    
    def build_hetero_adj(self, positions: torch.Tensor, r_comm: float) -> Dict:
        """
        per-edge-type binary adj for HetGAT layers
        returns dict with keys: scout_to_scout, scout_to_interc, etc.
        """
        n_s = self.n_scouts
        n_i = self.n_interceptors

        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        connected = (dist <= r_comm).float()

        # zero out self loops
        eye = torch.eye(n_s + n_i, device=positions.device).unsqueeze(0)
        connected = connected * (1.0 - eye)

        adj = {
            'scout_to_scout': connected[:, :n_s, :n_s],
            'scout_to_inter': connected[:, :n_s, n_s:],
            'inter_to_scout': connected[:, n_s:, :n_s],
            'inter_to_inter': connected[:, n_s:, n_s:]
        }

        batch = positions.shape[0]
        adj['scout_to_ssn'] = torch.ones(batch, n_s, 1, device=positions.device)
        adj['inter_to_ssn'] = torch.ones(batch, n_i, 1, device=positions.device)

        return adj
    








    


