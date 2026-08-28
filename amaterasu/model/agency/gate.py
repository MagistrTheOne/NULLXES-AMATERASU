from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.tensors.modality import GateDecision


class ConstraintGate(nn.Module):
    """ALLOW / DEFER / BLOCK. Input (d+32) → 1024 → 3, fp32."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model + 32, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 3, bias=False)

    def forward(self, intent: torch.Tensor, side32: torch.Tensor) -> torch.Tensor:
        b, k, _ = intent.shape
        side = side32.unsqueeze(1).expand(b, k, 32)
        x = torch.cat([intent.float(), side.float()], dim=-1)
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))

    @staticmethod
    def decide(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    @staticmethod
    def as_enum(idx: int) -> GateDecision:
        return GateDecision(int(idx))
