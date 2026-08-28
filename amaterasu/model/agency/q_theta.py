from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig


class QTheta(nn.Module):
    """Qθ(I | Z, A, G, pool): concat 4×d → d → 1. First linear has bias (ledger)."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=True)
        self.fc2 = nn.Linear(cfg.d_model, 1, bias=False)

    def forward(
        self,
        intent: torch.Tensor,
        z_pool: torch.Tensor,
        a_pool: torch.Tensor,
        g_pool: torch.Tensor,
    ) -> torch.Tensor:
        z = z_pool.unsqueeze(1).expand_as(intent)
        a = a_pool.unsqueeze(1).expand_as(intent)
        g = g_pool.unsqueeze(1).expand_as(intent)
        x = torch.cat([intent, z, a, g], dim=-1)
        return self.fc2(torch.nn.functional.silu(self.fc1(x.float()))).squeeze(-1)
