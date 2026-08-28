from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig


class TactileEncoder(nn.Module):
    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(cfg.tactile_in, cfg.d_model, bias=False)
        self.mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, tactile: torch.Tensor, tactile_valid: torch.Tensor) -> torch.Tensor:
        h = self.mix(self.in_proj(tactile))
        return (h * tactile_valid.to(h.dtype).unsqueeze(-1)).unsqueeze(1)
