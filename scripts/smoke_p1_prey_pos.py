import sys
sys.path.insert(0, '/Users/akshatguduru/Desktop/Thesis/gnn_experiments')
import torch
from environments.predator_prey import PredatorPreyAdapter
adapter = PredatorPreyAdapter(num_envs=256, device="cpu", max_steps=80, world_size=5.0)
adapter.hetnet_reset()
prey_positions = adapter.env.scenario.prey.state.pos
assert prey_positions.min().item() < -0.5, \
    f"prey_pos.min()={prey_positions.min()}, expected negative (should be in [-3, 3])"
assert prey_positions.max().item() > 0.5, f"prey_pos.max()={prey_positions.max()}"
print(f"PP prey_pos range: [{prey_positions.min():.2f}, {prey_positions.max():.2f}] -- PASS")
