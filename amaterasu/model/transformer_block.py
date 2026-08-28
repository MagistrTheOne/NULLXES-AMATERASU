from __future__ import annotations

import torch
from torch import nn

from amaterasu.model.attention.gqa import GQAAttention
from amaterasu.model.ffn.swiglu import SwiGLU
from amaterasu.model.norms.rmsnorm import RMSNorm


class PreNormBlock(nn.Module):
    """Pre-norm GQA + SwiGLU. Matches P_vel / HPT-width / dyn-width ledger layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
        eps: float = 1e-6,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model, eps=eps)
        self.attn = GQAAttention(
            d_model,
            n_heads,
            n_kv_heads,
            d_head,
            rope_theta=rope_theta,
            eps=eps,
            use_qk_norm=use_qk_norm,
        )
        self.norm_ffn = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
        kv_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm_attn(x)
        x = x + self.attn(h, position_ids, attn_bias=attn_bias, kv=kv, kv_position_ids=kv_position_ids)
        x = x + self.ffn(self.norm_ffn(x))
        return x
