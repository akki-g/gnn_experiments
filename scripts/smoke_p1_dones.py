import sys
sys.path.insert(0, '/Users/akshatguduru/Desktop/Thesis/gnn_experiments')
import torch
from environments.predator_prey import PredatorPreyAdapter
from environments.predator_capture_prey import PredatorCapturePreyAdapter

for name, AdapterCls, extra in [
    ("PP",  PredatorPreyAdapter, {}),
    ("PCP", PredatorCapturePreyAdapter, {}),
]:
    adapter = AdapterCls(num_envs=256, device="cpu", max_steps=80, **extra)
    adapter.hetnet_reset()
    dones_per_step = []
    trunc_per_step = []
    for t in range(256):
        acts = torch.randn(256, adapter.n_defenders, 2) * 0.1
        reward, done, info = adapter.hetnet_step(acts)
        dones_per_step.append(done.float().sum().item())
        trunc_per_step.append(info.get("is_truncation", torch.zeros_like(done)).float().sum().item())

    total_dones = sum(dones_per_step)
    total_trunc = sum(trunc_per_step)
    assert total_dones < 256 * 10, f"{name}: total_dones={total_dones}, expected ~768"
    assert total_dones > 256 * 2,  f"{name}: total_dones={total_dones}, expected ~768"
    for expected_t in [79, 159, 239]:
        assert dones_per_step[expected_t] >= 200, \
            f"{name}: dones at t={expected_t} = {dones_per_step[expected_t]}, expected ~256"
    assert abs(total_dones - total_trunc) < 256, \
        f"{name}: dones={total_dones} but trunc={total_trunc}"
    print(f"{name}: PASS  total_dones={total_dones:.0f}  total_trunc={total_trunc:.0f}")
