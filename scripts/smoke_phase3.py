import sys; sys.path.insert(0, '/Users/akshatguduru/Desktop/Thesis/gnn_experiments')
import torch
from environments.predator_prey import PredatorPreyAdapter
from gnn.trainer_multienv import GNNTrainerMultiEnv

adapter = PredatorPreyAdapter(num_envs=4, device='cpu', max_steps=80, discrete=True)
trainer = GNNTrainerMultiEnv(
    adapter=adapter, hidden_dim=32, lr=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, device='cpu',
    rollout_length=8, F_feat=16, G_feat=16, K=2, r_comm=1.5,
    discrete=True, n_actions=5
)
last_obs, metrics = trainer.collect_rollouts(num_steps=8)
print(f"GNN discrete rollout ok")
update = trainer.update(last_obs, num_actor_epochs=1, num_critic_epochs=1, B=4)
print(f"GNN discrete update ok, entropy={update.get('entropy', 0):.3f}")
print('Phase 3 PASS')
