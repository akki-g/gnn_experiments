from environments.grid_world import GridWorld

import torch


class PredatorPrey(GridWorld):
    """
    One class of 'n_predators' agents. Each must reach the single stationary prey
    Observation = own-location one-hot ++ FOV precence window
    """

    def __init__(self, n_predators: int = 3, **kwargs):
        super().__init__(classes=[("predator", n_predators, 5)], **kwargs)


    def _obs_dim(self, name: str) -> dict:
        return {"state": self.G * self.G, "obs": 3 * self.w * self.w}
    
    def _compute_obs(self) -> dict :
        sl = self._slices["predator"]
        loc = self._own_location_onehot(sl)
        fov = self._fov_windows(sl)
        return {"predator": {"state" : loc, "obs": fov}}
    
    def _compute_done(self, act: torch.Tensor) -> torch.Tensor:
        on_prey = (self.agent_pos == self.prey_pos[:, None, :]).all(-1)
        return on_prey