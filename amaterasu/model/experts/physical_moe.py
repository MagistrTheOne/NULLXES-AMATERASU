from __future__ import annotations

import torch
from torch import nn

from amaterasu.model.experts.metrics import expert_utilization, routing_entropy
from amaterasu.model.experts.router import SigmoidTopKRouter
from amaterasu.model.ffn.swiglu import SwiGLU


class PhysicalMoE(nn.Module):
    """8 routed + 1 shared SwiGLU experts. Real sparse dispatch: expert e runs only on selected tokens."""

    def __init__(self, d_model: int, d_ff_expert: int, n_routed: int, topk: int) -> None:
        super().__init__()
        self.n_routed = n_routed
        self.topk = topk
        self.router = SigmoidTopKRouter(d_model, n_routed, topk)
        self.routed = nn.ModuleList([SwiGLU(d_model, d_ff_expert) for _ in range(n_routed)])
        self.shared = SwiGLU(d_model, d_ff_expert)
        self.last_utilization: torch.Tensor | None = None
        self.last_entropy: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Return residual delta [B, S, D]. Non-masked positions stay zero."""
        b, s, d = x.shape
        delta = x.new_zeros(b, s, d)
        flat_mask = token_mask.reshape(-1)
        if not bool(flat_mask.any()):
            return delta
        idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        tokens = x.reshape(b * s, d).index_select(0, idx)
        topv, topi = self.router(tokens)
        self.last_utilization = expert_utilization(topi, self.n_routed)
        self.last_entropy = routing_entropy(topv)
        routed_out = tokens.new_zeros(tokens.shape)
        for e, expert in enumerate(self.routed):
            hit = topi == e
            if not bool(hit.any()):
                continue
            token_sel = hit.any(dim=-1)
            sel_idx = token_sel.nonzero(as_tuple=False).squeeze(-1)
            x_e = tokens.index_select(0, sel_idx)
            y_e = expert(x_e)
            w = torch.zeros(sel_idx.shape[0], device=tokens.device, dtype=tokens.dtype)
            picked = topi.index_select(0, sel_idx)
            weights = topv.index_select(0, sel_idx).to(tokens.dtype)
            for k in range(self.topk):
                pick = picked[:, k] == e
                w = torch.where(pick, weights[:, k], w)
            routed_out.index_add_(0, sel_idx, y_e * w.unsqueeze(-1))
        y = routed_out + self.shared(tokens)
        delta.reshape(b * s, d).index_copy_(0, idx, y)
        return delta
