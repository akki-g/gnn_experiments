import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from environments.predator_prey import PredatorPreyAdapter
from baseline.trainer import IPPOTrainer

# Discrete mode
adapter = PredatorPreyAdapter(num_envs=4, device='cpu', max_steps=80, discrete=True)
trainer = IPPOTrainer(
    adapter=adapter, hidden_dim=32, lr=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, device='cpu',
    discrete=True, n_actions=5
)
last_obs, metrics = trainer.collect_rollouts(num_steps=8)
print(f"Discrete rollout ok, mean_ret={metrics.get('mean_episode_return', 'N/A')}")
update = trainer.update(last_obs, num_actor_epochs=1, num_critic_epochs=1, B=4)
print(f"Discrete update ok, entropy={update.get('entropy', 'N/A'):.3f}")

# Continuous mode
adapter2 = PredatorPreyAdapter(num_envs=4, device='cpu', max_steps=80, discrete=False)
trainer2 = IPPOTrainer(
    adapter=adapter2, hidden_dim=32, lr=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, device='cpu',
    discrete=False
)
last_obs2, metrics2 = trainer2.collect_rollouts(num_steps=8)
print(f"Continuous rollout ok, mean_ret={metrics2.get('mean_episode_return', 'N/A')}")
print('Phase 2 PASS')
