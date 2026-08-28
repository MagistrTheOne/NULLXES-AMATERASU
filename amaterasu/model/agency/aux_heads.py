from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig

AUX_NAMES = (
    "uncertainty",
    "novelty",
    "social_relevance",
    "env_change",
    "intervention_value",
    "action_cost",
    "physical_risk",
    "persistence",
    "inhibition",
)


class AuxHeads(nn.Module):
    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.names = AUX_NAMES
        self.up = nn.ModuleList([nn.Linear(cfg.d_model, 512, bias=False) for _ in range(cfg.eac_aux_heads)])
        self.down = nn.ModuleList([nn.Linear(512, 1, bias=False) for _ in range(cfg.eac_aux_heads)])

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        outs = [down(torch.nn.functional.silu(up(pooled))) for up, down in zip(self.up, self.down)]
        return torch.cat(outs, dim=-1)
