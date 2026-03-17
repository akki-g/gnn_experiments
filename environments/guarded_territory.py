import typing
import torch
import vmas
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

# Agent Type definitions
SCOUT = "scout"
INTERCEPTOR = "interceptor"
INTRUDER = "intruder"


class Scenario(BaseScenario):
    """
    Guarded Territory: heterogeneous cooperative-competitive MARL scenario.

    Designed for GNN communication ablations with:
    - Heterogeneous observation spaces (scouts see further, interceptors see closer)
    - Communication constraints (GNN is the only inter-agent channel)
    - Mixed cooperative-competitive dynamics (defenders cooperate, intruders compete)

    Key Principle: Interceptors CANNOT succeed without scout communication.
    Ensures that r_comm and K ablations produce strong, interpretable signals.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Agent / World Config
        self.n_scouts = kwargs.get("n_scouts", 3)
        self.n_interceptors = kwargs.get("n_interceptors", 3)
        self.n_intruders = kwargs.get("n_intruders", 3)
        self.n_zones = kwargs.get("n_zones", 2)
        self.world_size = kwargs.get("world_size", 5.0)

        # Observation Radii - core asymmetry
        self.scout_fov = kwargs.get("scout_fov", 1.0)
        self.interceptor_fov = kwargs.get("interceptor_fov", 0.5)
        self.tag_radius = kwargs.get("tag_radius", 0.1)

        # Speeds
        self.intruder_speed = kwargs.get("intruder_speed", 0.5)
        self.defender_speed = kwargs.get("defender_speed", 0.8)

        self.n_defenders = self.n_scouts + self.n_interceptors

        self.intruder_skill = kwargs.get("intruder_skill", 0.0)

        # Create World
        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=0.1,
            drag=0.25,
            dim_c=0,  # no built-in comms - GNN handles this
            x_semidim=self.world_size,
            y_semidim=self.world_size,
        )

        # Create Scouts
        self.scouts = []
        for i in range(self.n_scouts):
            agent = Agent(
                name=f"scout_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.075),
                max_speed=self.defender_speed,
                color=Color.BLUE,
                u_range=1.0,
            )
            agent.agent_type = SCOUT
            agent.type_id = 0
            world.add_agent(agent=agent)
            self.scouts.append(agent)

        # Create Interceptors  [FIX: consistent spelling]
        self.interceptors = []
        for i in range(self.n_interceptors):
            agent = Agent(
                name=f"interceptor_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.09),
                max_speed=self.defender_speed,
                color=Color.GREEN,
                u_range=1.0,
            )
            agent.agent_type = INTERCEPTOR
            agent.type_id = 1
            world.add_agent(agent)
            self.interceptors.append(agent)

        # Create Intruders (scripted via process_action)
        # [FIX: removed action_script parameter - process_action handles everything]
        self.intruders = []
        for i in range(self.n_intruders):
            intruder = Agent(
                name=f"intruder_{i}",
                collide=True,
                mass=1.0,
                shape=Sphere(radius=0.075),
                max_speed=self.intruder_speed,
                color=Color.RED,
                u_range=1.0,
            )
            intruder.agent_type = INTRUDER
            intruder.type_id = 2
            world.add_agent(intruder)
            self.intruders.append(intruder)

        self.defenders = self.scouts + self.interceptors

        # Create Target Zones / Landmarks
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

        # Tracking Tensors (allocated in reset)
        self._intruder_tagged = None
        self._zone_breached = None
        self._tag_count = None


        # cache constant tensors 
        self._scout_type_oh = torch.tensor([1.0, 0.0], device=device).unsqueeze(0) # (1,2)
        self._interceptor_type_oh = torch.tensor([0.0, 1.0], device=device).unsqueeze(0) # (1,2)

        return world

    def reset_world_at(self, env_index: typing.Optional[int] = None):
        batch = self.world.batch_dim
        device = self.world.device
        self._intruder_speed_var = self.intruder_speed * (0.8 + 0.4 * torch.rand(1).item())

        if env_index is None:
            self._intruder_tagged = torch.zeros(
                batch, self.n_intruders, dtype=torch.bool, device=device
            )
            self._zone_breached = torch.zeros(
                batch, self.n_zones, dtype=torch.bool, device=device
            )
            self._tag_count = torch.zeros(batch, dtype=torch.float32, device=device)
            self._prev_intruder_tagged = self._intruder_tagged.clone()
            self._prev_zone_breached = self._zone_breached.clone()
        else:
            self._intruder_tagged[env_index] = False
            self._zone_breached[env_index] = False
            self._tag_count[env_index] = 0.0
            self._prev_intruder_tagged[env_index] = False
            self._prev_zone_breached[env_index] = False

        if env_index is None:
            self._intruder_targets = torch.randint(
                0, self.n_zones, (batch, self.n_intruders), device=device
            )
        else:
            self._intruder_targets[env_index] = torch.randint(
                0, self.n_zones, (self.n_intruders,), device=device
            )
        # Spawn zones in inner region
        for i, zone in enumerate(self.zones):
            pos = torch.zeros(
                (1, 2) if env_index is not None else (batch, 2),
                dtype=torch.float32, device=device,
            )
            angle = 2 * torch.pi * i / self.n_zones
            radius = 0.3
            pos[..., 0] = radius * torch.cos(torch.tensor(angle))
            pos[..., 1] = radius * torch.sin(torch.tensor(angle))
            pos += 0.1 * torch.randn_like(pos)
            zone.set_pos(pos, batch_index=env_index)

        # Spawn defenders near zones (inner ring)
        for i, defender in enumerate(self.defenders):
            pos = torch.zeros(
                (1, 2) if env_index is not None else (batch, 2),
                dtype=torch.float32, device=device,
            )
            angle = 2 * torch.pi * i / self.n_defenders
            radius = 0.5 + 0.2 * torch.rand(pos.shape[0], 1, device=device)
            pos[..., 0:1] = radius * torch.cos(torch.tensor(angle))
            pos[..., 1:2] = radius * torch.sin(torch.tensor(angle))
            pos += 0.05 * torch.randn_like(pos)
            defender.set_pos(pos, batch_index=env_index)

        # Spawn intruders on outer edge
        for i, intruder in enumerate(self.intruders):
            pos = torch.zeros(
                (1, 2) if env_index is not None else (batch, 2),
                dtype=torch.float32, device=device,
            )
            angle = 2 * torch.pi * torch.rand(1, device=device).item()
            radius = self.world_size * (0.7 + 0.2 * torch.rand(1, device=device).item())
            pos[..., 0] = radius * torch.cos(torch.tensor(angle))
            pos[..., 1] = radius * torch.sin(torch.tensor(angle))
            intruder.set_pos(pos, batch_index=env_index)

    def _get_intruder_actions(self, intruder: Agent) -> torch.Tensor:
      if not hasattr(self, '_intruder_speed_var'):
          self._intruder_speed_var = self.intruder_speed

      d, b = self.world.device, self.world.batch_dim

      # -- Skilled component (existing goal-seeking + evasion) --
      intruder_idx = self.intruders.index(intruder)
      target_zone = self.zones[intruder_idx % self.n_zones]
      to_target = target_zone.state.pos - intruder.state.pos
      dist_to_target = torch.linalg.vector_norm(to_target, dim=-1, keepdim=True) + 1e-6
      dir_target = to_target / dist_to_target

      evasion_radius = 0.6

      interc_pos = torch.stack([i.state.pos for i in self.interceptors], dim=0) 
      away = intruder.state.pos - interc_pos
      ic_dists = torch.linalg.vector_norm(away, dim=-1, keepdim=True) + 1e-6

      min_idx = ic_dists.squeeze(-1).argmin(dim=0)
      min_ic_dist = ic_dists.squeeze(-1).min(dim=0, keepdim=True).values.T

      # gather evasion dir for closes interc
      min_idx_exp = min_idx.unsqueeze(0).unsqueeze(-1).expand(1,-1,2)
      evade_dir = away.gather(0, min_idx_exp).squeeze(0) / min_ic_dist

      evade_weight = torch.clamp(1.0 - min_ic_dist / evasion_radius, min=0.0, max=0.8)
      goal_weight = 1.0 - evade_weight

      skilled_dir = goal_weight * dir_target + evade_weight * evade_dir
      skilled_dir = skilled_dir / (torch.linalg.vector_norm(skilled_dir, dim=-1, keepdim=True) + 1e-6)

      # -- Random component --
      random_dir = torch.randn(b, 2, device=d)
      random_dir = random_dir / (torch.linalg.vector_norm(random_dir, dim=-1, keepdim=True) + 1e-6)

      # -- Blend by skill level --
      skill = self.intruder_skill
      direction = skill * skilled_dir + (1 - skill) * random_dir
      direction = direction / (torch.linalg.vector_norm(direction, dim=-1, keepdim=True) + 1e-6)

      noise = 0.15 * torch.randn(b, 2, device=d)
      action = self._intruder_speed_var * (direction + noise)
      return action

    def process_action(self, agent: Agent):
        """Override to inject scripted actions for intruders."""
        if hasattr(agent, "agent_type") and agent.agent_type == INTRUDER:
            agent.action.u = self._get_intruder_actions(agent)

    def observation(self, agent: Agent) -> torch.Tensor:
        batch = self.world.batch_dim
        device = self.world.device

        if hasattr(agent, "agent_type") and agent.agent_type == SCOUT:
            fov = self.scout_fov
            type_oh = self._scout_type_oh.expand(batch, 2)
        elif hasattr(agent, "agent_type") and agent.agent_type == INTERCEPTOR:
            fov = self.interceptor_fov
            type_oh = self._interceptor_type_oh.expand(batch,2)
        else:
            return torch.zeros(batch, 2, device=device)

        obs_parts = []
        obs_parts.append(agent.state.vel)
        obs_parts.append(agent.state.pos)
        obs_parts.append(type_oh)

        zone_pos = torch.stack([z.state.pos for z in self.zones], dim=1)
        zone_rel = zone_pos - agent.state.pos.unsqueeze(1)
        obs_parts.append(zone_rel.reshape(batch, -1))

        intrud_pos = torch.stack([i.state.pos for i in self.intruders], dim=1)
        intrud_vel = torch.stack([i.state.vel for i in self.intruders], dim=1)
        intrud_rel = intrud_pos - agent.state.pos.unsqueeze(1)
        intrud_dist = torch.linalg.vector_norm(intrud_rel, dim=-1, keepdim=True)
        visible = (intrud_dist <= fov).float()
        obs_parts.append((intrud_rel * visible).reshape(batch, -1))
        obs_parts.append((intrud_vel*visible).reshape(batch, -1))
        
        other_pos = torch.stack(
            [d.state.pos for d in self.defenders if d is not agent], dim=1
        )
        other_rel = other_pos - agent.state.pos.unsqueeze(1)
        other_dist = torch.linalg.vector_norm(other_rel, dim=-1, keepdim=True)
        other_vis = (other_dist <= fov).float()
        obs_parts.append((other_rel * other_vis).reshape(batch, -1))

        return torch.cat(obs_parts, dim=-1)

    def pre_step(self):
        if self._intruder_tagged is None:
            return

        self._prev_intruder_tagged = self._intruder_tagged.clone()
        self._prev_zone_breached = self._zone_breached.clone()

        # stack all pos
        interc_pos = torch.stack([i.state.pos for i in self.interceptors], dim=0) # (I, B, 2)
        intrud_pos = torch.stack([i.state.pos for i in self.intruders], dim=0) # (J, B, 2)

        dists = torch.linalg.vector_norm(
            interc_pos.unsqueeze(1) - intrud_pos.unsqueeze(0), dim=-1
        ) # (I, J, B)

        tagged_now = (dists < self.tag_radius).any(dim=0)
        tagged_now = tagged_now.permute(1,0) # (J,B) -> (B,J)

        new_tags = tagged_now & (~self._intruder_tagged)
        self._intruder_tagged = self._intruder_tagged | tagged_now
        self._tag_count += new_tags.float().sum(dim=-1)

        # zone breach: stack zone pos (K,B,2) intruder positions (J,B,2)
        zone_pos = torch.stack([z.state.pos for z in self.zones], dim=0)
        zone_dists = torch.linalg.vector_norm(
            zone_pos.unsqueeze(1) - intrud_pos.unsqueeze(0), dim=-1
        )

        # reduce over j
        not_tagged = (~self._intruder_tagged).permute(1,0)
        breached_any = ((zone_dists <0.15) & not_tagged.unsqueeze(0)).any(dim=1)
        breached_any = breached_any.permute(1,0)
        self._zone_breached = self._zone_breached | breached_any



    def reward(self, agent: Agent) -> torch.Tensor:
        if hasattr(agent, "agent_type") and agent.agent_type == INTRUDER:
            return torch.zeros(self.world.batch_dim, device=self.world.device)

        batch = self.world.batch_dim
        device = self.world.device
        rew = torch.zeros(batch, device=device)

        new_tags = self._intruder_tagged & (~self._prev_intruder_tagged)
        n_new_tags = new_tags.float().sum(dim=-1)
        rew += 3.0 * n_new_tags

        new_breaches = self._zone_breached & (~self._prev_zone_breached)
        n_new_breaches = new_breaches.float().sum(dim=-1)
        rew -= 5.0 * n_new_breaches

        # Individual shaping
        if hasattr(agent, "agent_type") and agent.agent_type == SCOUT:
            for intruder in self.intruders:
                dist_to_intruder = torch.linalg.vector_norm(
                    agent.state.pos - intruder.state.pos, dim=-1
                )
                sees_intruder = (dist_to_intruder < self.scout_fov).float()
                rew += 0.5 * sees_intruder

                for interceptor in self.interceptors:
                    dist_to_interceptor = torch.linalg.vector_norm(
                        agent.state.pos - interceptor.state.pos, dim=-1
                    )
                    in_relay_range = ((dist_to_interceptor > 0.2) & (dist_to_interceptor < 1.0)).float()
                    rew += 0.2 * sees_intruder * in_relay_range

        elif hasattr(agent, "agent_type") and agent.agent_type == INTERCEPTOR:
            min_dist = torch.full((batch,), float("inf"), device=device)
            best_closing_speed = torch.zeros(batch, device=device)
            for j, intruder in enumerate(self.intruders):
                if self._intruder_tagged is not None:
                    tagged = self._intruder_tagged[:, j]
                else:
                    tagged = torch.zeros(batch, dtype=torch.bool, device=device)

                rel_pos = intruder.state.pos - agent.state.pos
                dist = torch.linalg.vector_norm(rel_pos, dim=-1)
                # closing speed
                rel_vel = agent.state.vel - intruder.state.vel
                closing = -(rel_vel * rel_pos).sum(dim=-1) / (dist + 1e-6)

                effective_dist = torch.where(
                    tagged, torch.tensor(float('inf'), device=device), dist
                )
                closer_mask = effective_dist < min_dist
                min_dist = torch.where(closer_mask, effective_dist, min_dist)
                best_closing_speed = torch.where(closer_mask, closing, best_closing_speed)

            min_dist = torch.clamp(min_dist, max=5.0)
            rew -= 0.5 * min_dist # penalize dist
            rew += 0.3 * best_closing_speed.clamp(-1, 1)

            # proximity bonus
            near_tag = torch.clamp(1.0 - min_dist / (self.tag_radius * 3), min=0)
            rew += 2.0 * near_tag
        return rew

    def done(self) -> torch.Tensor:
        if self._intruder_tagged is None or self._zone_breached is None:
            return torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            )
        all_tagged = self._intruder_tagged.all(dim=-1)
        all_breached = self._zone_breached.all(dim=-1)
        return all_tagged | all_breached

    def info(self, agent: Agent) -> dict:
        info = {}
        if self._intruder_tagged is not None:
            info["n_tagged"] = self._intruder_tagged.sum(dim=-1).float()
        if self._zone_breached is not None:
            info["n_breached"] = self._zone_breached.sum(dim=-1).float()
        return info


def get_obs_dim(n_scouts=3, n_interceptors=3, n_intruders=3, n_zones=2):
    n_defenders = n_scouts + n_interceptors
    return (
        2 + 2 + 2                  # vel + pos + type_one_hot
        + n_zones * 2              # zone relative positions
        + n_intruders * 2          # intruder rel pos (masked)
        + n_intruders * 2          # intruder vel (masked)
        + (n_defenders - 1) * 2    # other defenders rel pos
    )




class GuardedTerritoryAdapter:
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        n_scouts: int = 3,
        n_interceptors: int = 3,
        n_intruders: int = 3,
        n_zones: int = 2,
        max_steps: int = 200,
        **kwargs,
    ):
        self.num_envs = num_envs
        self.device = device
        self.n_scouts = n_scouts
        self.n_interceptors = n_interceptors
        self.n_intruders = n_intruders
        self.n_defenders = n_scouts + n_interceptors
        self.n_zones = n_zones
        
        self.env = vmas.make_env(
            scenario=Scenario(),
            num_envs=num_envs,
            device=device,
            continuous_actions=True,
            max_steps=max_steps,
            n_scouts=n_scouts,
            n_interceptors=n_interceptors,
            n_intruders=n_intruders,
            n_zones=n_zones,
            **kwargs,
        )

        self.obs_dim = get_obs_dim(n_scouts, n_interceptors, n_intruders, n_zones)

        self.defender_indices = []
        self.intruder_indices = []
        for i, agent in enumerate(self.env.agents):
            if hasattr(agent, "agent_type"):
                if agent.agent_type in (SCOUT, INTERCEPTOR):
                    self.defender_indices.append(i)
                elif agent.agent_type == INTRUDER:
                    self.intruder_indices.append(i)

        self._defender_idx = torch.tensor(self.defender_indices, device=device)
        self._intruder_idx = torch.tensor(self.intruder_indices, device=device)
        self._n_agents_total = len(self.env.agents)
        self._action_dim = 2
        
        assert len(self.defender_indices) == self.n_defenders

        self.agent_types = []
        for idx in self.defender_indices:
            self.agent_types.append(self.env.agents[idx].type_id)
        self.agent_types = torch.tensor(self.agent_types, device=device)

    def reset(self):
        """Returns obs (num_envs, n_def, obs_dim), positions (num_envs, n_def, 2)"""
        all_obs = self.env.reset()
        defender_obs = torch.stack([all_obs[i] for i in self.defender_indices], dim=1)
        positions = defender_obs[:, :, 2:4].clone()
        return defender_obs, positions

    def step(self, defender_actions: torch.Tensor):
        """
        Args: defender_actions (num_envs, n_defenders, 2)
        Returns: obs, rewards, dones, info, positions
        """
        # pre allocate full action tensor on device
        all_actions_tensor = torch.zeros(
            self.num_envs, self._n_agents_total, self._action_dim,
            device=self.device
        )

        # scatter defender actions to right agent slots
        all_actions_tensor[:, self._defender_idx] = defender_actions

        all_actions = [all_actions_tensor[:, i] for i in range(self._n_agents_total)]

        all_obs, all_rewards, dones, all_infos = self.env.step(all_actions)

        defender_obs = torch.stack([all_obs[i] for i in self.defender_indices], dim=1)
        defender_rewards = torch.stack([all_rewards[i] for i in self.defender_indices], dim=1)
        positions = defender_obs[:, :, 2:4].clone()

        info = {}
        if len(all_infos) > 0 and self.defender_indices:
            first_info = all_infos[self.defender_indices[0]]
            if isinstance(first_info, dict):
                info = first_info
        
        return defender_obs, defender_rewards, dones, info, positions

    def build_adj(self, positions: torch.Tensor, r_comm: float) -> torch.Tensor:
        """Build row-normalized adj matrix. positions: (num_envs, n_def, 2)"""
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        adj = (dist <= r_comm).float()
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / deg
        return adj
    def reset_env(self):
      """Full manual reset - returns obs (num_envs, n_def, obs_dim), positions."""
      all_obs = self.env.reset()
      defender_obs = torch.stack([all_obs[i] for i in self.defender_indices], dim=1)
      positions = defender_obs[:, :, 2:4].clone()
      return defender_obs, positions

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def n_agents(self) -> int:
        return self.n_defenders
