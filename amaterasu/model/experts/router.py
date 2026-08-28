from __future__ import annotations

import torch
from torch import nn


class SigmoidTopKRouter(nn.Module):
    """Physical MoE router. Logits in fp32. Sigmoid then top-k, then renormalize."""

    def __init__(self, d_model: int, n_experts: int, topk: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.topk = topk
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts), persistent=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x.float()) + self.expert_bias.float()
        scores = torch.sigmoid(logits)
        topv, topi = scores.topk(self.topk, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return topv, topi
