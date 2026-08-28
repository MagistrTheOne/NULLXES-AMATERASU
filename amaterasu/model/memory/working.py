from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.attention.gqa import GQAAttention


class WorkingMemory(nn.Module):
    """256 learned slots, one GQA cross-attn (P_attn, no QK-norm), write Linear(d,d,bias=True)."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.zeros(cfg.wm_slots, cfg.d_model))
        self.attn = GQAAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.d_head,
            rope_theta=cfg.rope_theta,
            eps=cfg.rms_eps,
            use_qk_norm=False,
        )
        self.write = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def forward(self, context: torch.Tensor, context_pos: torch.Tensor) -> torch.Tensor:
        b = context.shape[0]
        slots = self.slots.unsqueeze(0).expand(b, -1, -1)
        q_pos = torch.arange(slots.shape[1], device=context.device).view(1, -1).expand(b, -1)
        read = self.attn(slots, q_pos, kv=context, kv_position_ids=context_pos)
        return slots + self.write(read)
