"""
Utility functions for the MAPPO backbone.

Implemented (real bodies): set_seed, load_yaml_config, get_device.
Stubbed (Step 2):          grad_norm, to_device, make_optimizer.
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml


# ---------------------------------------------------------------------------
# Fully implemented helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Sets: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all devices),
    and enables cuDNN deterministic mode (disables benchmark selection).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml_config(path: str) -> dict:
    """
    Load a YAML config file and return it as a plain dict.

    Parameters
    ----------
    path : absolute or relative path to the .yaml file.

    Returns
    -------
    config : dict (may be nested).
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Return the best available device.

    Parameters
    ----------
    prefer_cuda : if True and CUDA is available, return the first CUDA device;
                  otherwise return CPU.

    Returns
    -------
    device : torch.device("cuda:0") or torch.device("cpu").
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stubbed helpers (Step 2)
# ---------------------------------------------------------------------------

def grad_norm(parameters) -> float:
    """
    Compute the total gradient norm across all parameters.

    Parameters
    ----------
    parameters : iterable of Tensors (e.g. model.parameters()).

    Returns
    -------
    total_norm : float.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return 0.0
    return torch.norm(
        torch.stack([p.grad.detach().data.norm(2) for p in params_with_grad]), 2
    ).item()


def to_device(batch: Any, device: torch.device) -> Any:
    """
    Recursively move a dict/list/tuple of Tensors to `device`.

    Parameters
    ----------
    batch  : dict, list, tuple, or Tensor.
    device : target device.

    Returns
    -------
    Same structure with all Tensors moved to device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    else:
        return batch


def make_optimizer(
    params,
    lr: float,
    eps: float = 1.0e-5,
) -> torch.optim.Optimizer:
    """
    Construct a single joint Adam optimizer (decision 3: one optimizer for
    encoder + actor + critic).

    Parameters
    ----------
    params : iterable of parameters (e.g. model.parameters()).
    lr     : learning rate.
    eps    : Adam epsilon for numerical stability.

    Returns
    -------
    optimizer : torch.optim.Adam instance.
    """
    return torch.optim.Adam(list(params), lr=lr, eps=eps)
