from __future__ import annotations

import torch
from torch import nn

from amaterasu.model.norms.rmsnorm import RMSNorm


class QKNorm(nn.Module):
    """Shared-across-heads QK RMSNorm. Parameter count = 2 * d_head."""

    def __init__(self, d_head: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.q_norm = RMSNorm(d_head, eps=eps)
        self.k_norm = RMSNorm(d_head, eps=eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)
