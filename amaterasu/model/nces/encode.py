from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.transformer_block import PreNormBlock


class NCESEncoder(nn.Module):
    """6 VE-width layers. Input 128-d node features. Missing nodes masked, not zero-as-valid."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(cfg.nces_in, cfg.nces_d, bias=False)
        self.layers = nn.ModuleList(
            [
                PreNormBlock(
                    cfg.nces_d,
                    cfg.ve_heads,
                    cfg.ve_kv,
                    cfg.ve_d_head,
                    cfg.ve_d_ff,
                    rope_theta=cfg.rope_theta,
                    eps=cfg.rms_eps,
                )
                for _ in range(cfg.nces_layers)
            ]
        )
        self.out_proj = nn.Linear(cfg.nces_d, cfg.d_model, bias=False)

    def forward(self, nces_feat: torch.Tensor, nces_valid: torch.Tensor) -> torch.Tensor:
        b, n, _ = nces_feat.shape
        x = self.in_proj(nces_feat)
        x = x * nces_valid.unsqueeze(-1).to(x.dtype)
        pos = torch.arange(n, device=nces_feat.device).view(1, n).expand(b, n)
        valid_bias = (~nces_valid).unsqueeze(1).unsqueeze(1)
        attn_bias = torch.zeros(b, 1, n, n, device=nces_feat.device, dtype=torch.float32)
        attn_bias = attn_bias.masked_fill(valid_bias, torch.finfo(torch.float32).min)
        for layer in self.layers:
            x = layer(x, pos, attn_bias=attn_bias)
        x = self.out_proj(x)
        return x * nces_valid.unsqueeze(-1).to(x.dtype)
