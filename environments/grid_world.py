from __future__ import annotations

import torch
import torch.nn.functional as F

# canonical aciton indicies
STAY, UP, DOWN, LEFT, RIGHT, CAPTURE = 0, 1, 2, 3, 4, 5

# movement deltas in form (row, col)

_DELTAS = torch.tensor(
    [[0,0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=torch.long
    )

class GridWorld: 
    """
    Base vectorized grid world 
    holds shared mechanics like:
    movemnet + boundary clamping, occupancy/fov window extraction, 
    step penalty, episode term/trun, and resetting

    sub classes define the agent classes and class specific obs schemes
    """

    def __init__(
            self,
            classes,
            num_envs: int = 64,
            grid_size: int = 10,
            vision: int = 1, #fov radius (2* vision+1)^2 window
            max_steps: int = 80,
            step_penalty: float = -0.05,
            see_teammates: bool = True,
            device: str = "cpu", 
            seed: int | None = None
            ):
        
        self.device = torch.device(device)
        self.B = num_envs
        self.G = grid_size
        self.v = vision 
        self.w = 2 * vision + 1
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.see_teammates = see_teammates

        # agent class bookkeeping
        self.class_names = [c[0] for c in classes]
        self.class_counts = {c[0]: c[1] for c in classes}
        self.class_n_actions = {c[0]: c[2] for c in classes}
        self.N = sum(c[1] for c in classes)

        #flat agent ordering: [class0 agents..., class1 agents.,,]
        #agent_class_id[n] = index into self.class_names for agent n
        self._slices = {}       #class_name -> slice over the agent dim
        agent_class_id = [] 
        cursor = 0

        for name, count, _ in classes:
            self._slices[name] = slice(cursor, cursor + count)
            agent_class_id += [self.class_names.index(name)] * count 
            cursor += count

        self.agent_class_id = torch.tensor(agent_class_id, device=self.device)

        # boolean mask per class over the agent dim for vectorized ops

        self._class_mask = {
            name: (self.agent_class_id == self.class_names.index(name))
            for name in self.class_names
        }

        self._deltas = _DELTAS.to(self.device)
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)


        # state tensors 
        self.agent_pos = None       # (B, 2)    long
        self.prey_pos = None        # (B, 2)    long
        self.t = None               # (B, )     long
        self.agent_done = None      # (B, N)    Bool, latched obj completion
        

    def reset(self):
        self.agent_pos = torch.zeros(self.B, self.N, 2, dtype=torch.long, device=self.device)
        self.prey_pos = torch.zeros(self.B, 2, dtype=torch.long, device=self.device) 
        self.t = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.agent_done = torch.zeros(self.B, self.N, dtype=torch.bool, device=self.device)
        self._reset_idx(torch.ones(self.B, dtype=torch.bool, device=self.device))
        return self._compute_obs()
    
    def _reset_idx(self, mask: torch.Tensor):
        # re-init the envs selected by the mask
        n = int(mask.sum())
        if n == 0:
            return
        

        G = self.G

        # random agent pos 
        pos = torch.randint(0, G, (n, self.N, 2), generator=self._rng, device=self.device)

        # random prey pos, resample so it doesnt start at an agent
        prey = torch.randint(0, G, (n, 2), generator=self._rng, device=self.device)
        coincide = (pos == prey[:, None, :]).all(-1).any(-1)
        tries = 0
        while bool(coincide.any()) and tries < 20:
            k = int(coincide.sum())
            prey[coincide] = torch.randint(0, G, (k, 2), generator=self._rng, device=self.device)
            coincide = (pos == prey[:, None, :]).all(-1).any(-1)
            tries += 1
        
        self.agent_pos[mask] = pos
        self.prey_pos[mask] = prey
        self.t[mask] = 0
        self.agent_done[mask] = False

    def step(self, actions: dict):
        act = self._flatten_actions(actions)    # (B, N)

        # done agents are frozen: so force them to stay in current pos
        act = torch.where(self.agent_done, torch.full_like(act, STAY), act)

        # movement (clamp to grid) 
        delta = self._deltas[act]
        self.agent_pos = (self.agent_pos + delta).clamp_(0, self.G - 1)
        
        # obj completion 
        newly_done = self._compute_done(act)
        self.agent_done = self.agent_done | newly_done

        # reward
        reward = torch.where(
            self.agent_done,
            torch.zeros(self.B, self.N, device=self.device),
            torch.full((self.B, self.N), self.step_penalty, device=self.device),
        )


        # term/trunc
        terminated = self.agent_done.all(dim=1)
        self.t += 1
        truncated = self.t >= self.max_steps
        done = terminated | truncated

        # obs before auto reset (true terminal obs)
        obs = self._compute_obs()

        info = {
            "success": terminated.clone(), 
            "truncated": truncated.clone(),
            "episode_len": self.t.clone(),
            "final_obs": obs,
        }

        if bool(done.any()):
            self._reset_idx(done)
            obs = self._compute_obs()

        return obs, reward, terminated, truncated, info
    
    # shared helpers   
    def _flatten_actions(self, actions: dict) -> torch.Tensor:
        act = torch.empty(self.B, self.N, dtype=torch.long, device=self.device)
        for name, sl in self._slices.items():
            act[:, sl] = actions[name].to(self.device)
        return act
    
    def _own_location_onehot(self, sl: slice) -> torch.Tensor:
        """(B, N_c, G*G) one-hot of each agent's own cell """
        pos = self.agent_pos[:, sl, :]
        idx = pos[..., 0] * self.G + pos[..., 1]
        return F.one_hot(idx, self.G*self.G).float()
    
    def _presence_grids(self) -> torch.Tensor:
        """
        Build (B, C, G, G) presence channels: [agents, prey, inside-marker]
        counts are kept here; self-subtraction + binarization happens per agent
        in _fov_windows so each agent sees *other* agents but not itself
        """

        B, G = self.B, self.G
        flat = torch.zeros(B, G*G, device=self.device)

        # agent occupany counts
        a_idx = self.agent_pos[..., 0] * G + self.agent_pos[..., 1]
        flat.scatter_add_(1, a_idx, torch.ones_like(a_idx, dtype=flat.dtype))
        agents = flat.view(B, 1, G, G)
        
        # prey occupancy counts (single prey)
        prey = torch.zeros(B, G*G, device=self.device)
        p_idx = (self.prey_pos[:, 0] * G + self.prey_pos[:, 1]).unsqueeze(1)
        prey.scatter_(1, p_idx, 1.0)
        prey = prey.view(B, 1, G, G)

        inside = torch.ones(B, 1, G, G, device=self.device)
        return torch.cat([agents, prey, inside], dim=1)
    
    def _fov_windows(self, sl: slice) -> torch.Tensor:
        """
        (B, N_c, C*w*w) FOV observation for class slice sl
        Channel order per cell: [other-agents, prey, outside]
        """
        B, G, v, w = self.B, self.G, self.v, self.w
        grids = self._presence_grids()

        # pad with 0: the inside-marker channel becomes the basis for "outside"
        padded = F.pad(grids, (v, v, v, v), value=0.0)      # (B, 3, G+2v, G+2v)    
        # convert inside-marer to outside indicator (1 outside grid, 0 inside)
        padded[:, 2] = 1.0 - padded[:, 2]

        # all sliding windows, then gather the one centered on each agent
        wins = padded.unfold(2, w, 1).unfold(3, w, 1)       # (B, 3, G, G, w, w)

        pos = self.agent_pos[:, sl, :]
        rows, cols = pos[..., 0], pos[..., 1]
        b_idx = torch.arange(B, device=self.device)[:, None].expand_as(rows)
        win = wins[b_idx, :, rows, cols]

        if self.see_teammates:
            # remove the agent itself from the center of the agent channel
            win[:, :, 0, v, v] = (win[:, :, 0, v, v] - 1).clamp_(min=0)
            win[:, :, 0] = (win[:, :, 0] > 0).float()   
        else:
            win[:, :, 0] = 0.0
        
        return win.reshape(B, pos.shape[1], -1)
    
    # metadata
    @property
    def obs_dim(self) -> dict:
        return {name: self._obs_dim(name) for name in self.class_names}
    
    @property
    def actions_dims(self) -> dict:
        return dict(self.class_n_actions)
    
    # hooks for subclasses
    def _compute_obs(self) -> dict:
        raise NotImplementedError
    
    def _compute_done(self, act: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def _obs_dim(self, name: str) -> int:
        raise NotImplementedError
    
     