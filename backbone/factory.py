"""
Checkpoint-aware agent factory for the probe + eval-sweep drivers.

The probe (`run_probe.py`) and eval-sweep (`scripts/pcp_eval_sweep.py`) only ever call
`encode()` and `act()` — both provided by `MAPPO` and `QAgent` — so the QMIX mixer is
NOT needed at probe/eval time. This factory loads either backbone from a checkpoint,
dispatching on the saved `config["algo"]` field. Legacy MAPPO checkpoints have no
`"algo"` key, so the default `"mappo"` keeps them working unchanged.
"""

import torch


def build_agent_from_ckpt(ckpt_path: str, comm_module, device_str: str):
    """Load a trained agent (MAPPO or QMIX) for evaluation/probing.

    Parameters
    ----------
    ckpt_path   : path to a checkpoint saved by MAPPO.save() or QAgent.save().
    comm_module : a freshly built CommModule (its weights are overwritten by the
                  checkpoint's comm state on load) — built by the caller from the
                  run's sibling config.yaml, exactly as before.
    device_str  : target device string; overrides the (training) device saved in the
                  checkpoint config so GPU-trained checkpoints load on CPU eval nodes.

    Returns an eval-mode module exposing encode()/act().
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    algo = cfg.pop("algo", "mappo")

    if algo == "qmix":
        from backbone.q_agent import QAgent
        agent = QAgent(
            obs_dim=cfg["obs_dim"],
            action_dim=cfg["action_dim"],
            num_agents=cfg["num_agents"],
            hidden_dim=cfg["hidden_dim"],
            comm_module=comm_module,
            encoder_layers=cfg.get("encoder_layers", 2),
            head_layers=cfg.get("head_layers", 1),
            state_dim=cfg.get("state_dim"),
            device=device_str,
        )
        agent.load(ckpt_path, map_location=torch.device(device_str))
        return agent.eval()

    # default: MAPPO (legacy checkpoints carry no "algo" key)
    from backbone.mappo import MAPPO
    cfg["device"] = device_str
    model = MAPPO(**cfg, comm_module=comm_module)
    model.load(ckpt_path, map_location=torch.device(device_str))
    return model.eval()
