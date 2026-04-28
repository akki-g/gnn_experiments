import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from environments.predator_prey import PredatorPreyAdapter
from HetGAT.hetnetPolicy import HetNetPolicy
from HetGAT.critic import HetNetCritic
from HetGAT.mahac import MAHAC
from HetGAT.rollout import MAHACBuffer
from HetGAT.utils import collect_rollout, build_ssn_input

device = torch.device('cpu')
n_scouts = 2
n_interc = 1
n_envs = 4
rollout_len = 8

adapter = PredatorPreyAdapter(num_envs=n_envs, device='cpu', max_steps=80, discrete=True)

obs_portion_dim = adapter.obs_portion_dim
state_dim = adapter.state_dim

policy = HetNetPolicy(
    obs_dim_scout=obs_portion_dim,
    obs_dim_interc=obs_portion_dim,
    state_dim_scout=state_dim,
    state_dim_interc=state_dim,
    action_dim=adapter.action_dim,
    hidden_dim=32,
    n_heads=2,
    head_dim=8,
    n_layers=2,
    ssn_input_dim=5,
    r_comm=1.5,
    discrete=True,
    n_actions=5,
).to(device)

critic = HetNetCritic(ssn_dim=2*8, mode='per_class').to(device)

trainer = MAHAC(
    policy=policy,
    critic=critic,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.25,
    ppo_epochs=1,
    log_std_min=-1.5,
    discrete=True,
)

buffer = MAHACBuffer(
    T=rollout_len,
    B=n_envs,
    n_scouts=n_scouts,
    n_interc=n_interc,
    obs_dim_scout=obs_portion_dim,
    obs_dim_interc=obs_portion_dim,
    state_dim_scout=state_dim,
    state_dim_interc=state_dim,
    action_dim=adapter.action_dim,
    ssn_dim=5,
    hidden_dim=32,
    device=device,
    discrete=True,
)

bootstrap_values = collect_rollout(adapter, policy, critic, buffer, n_scouts, n_interc)
print(f"Rollout complete, buffer ptr={buffer.ptr}")

metrics = trainer.update(buffer, bootstrap_values)
print(f"Update ok, entropy_scout={metrics['entropy_scout']:.3f}, entropy_interc={metrics['entropy_interc']:.3f}")
import math
assert math.isfinite(metrics['entropy_scout']), f"entropy_scout not finite: {metrics['entropy_scout']}"
assert math.isfinite(metrics['entropy_interc']), f"entropy_interc not finite: {metrics['entropy_interc']}"
print('Phase 4 PASS')
