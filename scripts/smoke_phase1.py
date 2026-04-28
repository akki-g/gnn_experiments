import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from environments.predator_prey import PredatorPreyAdapter
from environments.predator_capture_prey import PredatorCapturePreyAdapter

for AdapterCls in [PredatorPreyAdapter, PredatorCapturePreyAdapter]:
    name = AdapterCls.__name__
    # Continuous mode
    a = AdapterCls(num_envs=4, device='cpu', max_steps=80, discrete=False)
    obs, pos = a.reset()
    a.step(torch.randn(4, a.n_defenders, 2))
    print(f'{name} continuous: OK')
    # Discrete mode
    a = AdapterCls(num_envs=4, device='cpu', max_steps=80, discrete=True)
    assert a.discrete is True, 'discrete not set'
    assert a.n_actions == 5, f'n_actions={a.n_actions}'
    obs, pos = a.reset()
    a.step(torch.randint(0, 5, (4, a.n_defenders)))
    print(f'{name} discrete: OK, n_actions={a.n_actions}')
print('Phase 1 PASS')
