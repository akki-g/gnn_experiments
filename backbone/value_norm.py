import torch
import torch.nn as nn
from torch import Tensor


class ValueNorm(nn.Module):
    """PopArt-style debiased running normalizer for value targets.
    Buffers only -> NO trainable params; excluded from .parameters().
    Tracks running mean and mean-of-squares with EMA + debiasing term."""

    def __init__(self, input_dim: int = 1, beta: float = 0.99999, epsilon: float = 1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_mean_sq", torch.zeros(input_dim))
        self.register_buffer("debiasing_term", torch.zeros(1))

    def _debiased_mean_var(self):
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        var = (mean_sq - mean ** 2).clamp(min=1e-2)
        return mean, var

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        x = x.reshape(-1, self.input_dim)
        batch_mean = x.mean(dim=0)
        batch_mean_sq = (x ** 2).mean(dim=0)
        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

    def normalize(self, x: Tensor) -> Tensor:
        mean, var = self._debiased_mean_var()
        return (x - mean) / torch.sqrt(var + self.epsilon)

    def denormalize(self, x: Tensor) -> Tensor:
        mean, var = self._debiased_mean_var()
        return x * torch.sqrt(var + self.epsilon) + mean
