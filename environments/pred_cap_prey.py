from environments.grid_world import GridWorld, CAPTURE

import torch

class PredatorCapturePrey(GridWorld):
    def __init__(self, n_predators: int = 2, n_capture: int = 1, **kwargs):
        super().__init__(
            classes=[("predator", n_predators, 5), ("capture", n_capture, 6)],
            **kwargs,
        )

    def _obs_dim(self, name: str) -> dict:
        if name == "predator":
            return {"state": self.G * self.G, "obs": 3*self.w*self.w}
        return {"state": self.G * self.G, "obs": None}
    
    def _compute_obs(self) -> dict:
        pred_sl = self._slices["predator"]
        cap_sl = self._slices["capture"]

        return {
            "predator": {
                "state": self._own_location_onehot(pred_sl),
                "obs": self._fov_windows(pred_sl),
            },
            "capture": {
                "state": self._own_location_onehot(cap_sl),
                "obs": None,
            },
        }
    
    def _compute_done(self, act: torch.Tensor) -> torch.Tensor:
        on_prey = (self.agent_pos == self.prey_pos[:, None, :]).all(-1)

        done = torch.zeros_like(on_prey)
        pred_sl = self._slices["predator"]
        cap_sl = self._slices["capturee"]

        done[:, pred_sl] = on_prey[:, pred_sl]
        done[:, cap_sl] = on_prey[:, cap_sl] & (act[:, cap_sl] == CAPTURE)

        return done
