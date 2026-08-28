from __future__ import annotations

import torch


def expert_utilization(topi: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Fraction of tokens that selected each expert. topi: [N, K]."""
    counts = torch.zeros(n_experts, device=topi.device, dtype=torch.float32)
    ones = torch.ones(topi.numel(), device=topi.device, dtype=torch.float32)
    counts.scatter_add_(0, topi.reshape(-1).long(), ones)
    denom = max(int(topi.shape[0]), 1)
    return counts / float(denom)


def routing_entropy(topv: torch.Tensor) -> torch.Tensor:
    p = topv.float().clamp_min(1e-9)
    p = p / p.sum(dim=-1, keepdim=True)
    return -(p * p.log()).sum(dim=-1).mean()
