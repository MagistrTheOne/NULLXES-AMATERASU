from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from amaterasu.model.attention.rope import apply_rope
from amaterasu.model.norms.qk_norm import QKNorm


class GQAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        rope_theta: float = 10_000.0,
        eps: float = 1e-6,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if n_heads * d_head != d_model:
            raise ValueError("n_heads * d_head must equal d_model")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.repeats = n_heads // n_kv_heads
        self.rope_theta = rope_theta
        self.use_qk_norm = use_qk_norm
        self.wq = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.wo = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.qk_norm = QKNorm(d_head, eps=eps) if use_qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
        kv_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        if s == 0:
            return x
        src = x if kv is None else kv
        sk = src.shape[1]
        q = self.wq(x).view(b, s, self.n_heads, self.d_head)
        k = self.wk(src).view(b, sk, self.n_kv_heads, self.d_head)
        v = self.wv(src).view(b, sk, self.n_kv_heads, self.d_head)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)
        q, k = apply_rope(q, k, position_ids, self.rope_theta, kv_position_ids)
        if self.repeats != 1:
            k = k.repeat_interleave(self.repeats, dim=2)
            v = v.repeat_interleave(self.repeats, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.wo(out)
