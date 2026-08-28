from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.transformer_block import PreNormBlock


class AudioEncoder(nn.Module):
    """8-layer d=768 MHA (not GQA) + SwiGLU 2048. No QK-norm (ledger)."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.frontend = nn.Linear(cfg.audio_mel, cfg.audio_d, bias=False)
        self.layers = nn.ModuleList(
            [
                PreNormBlock(
                    cfg.audio_d,
                    cfg.audio_heads,
                    cfg.audio_heads,
                    cfg.audio_d_head,
                    cfg.audio_d_ff,
                    rope_theta=cfg.rope_theta,
                    eps=cfg.rms_eps,
                    use_qk_norm=False,
                )
                for _ in range(cfg.audio_layers)
            ]
        )
        self.proj = nn.Linear(cfg.audio_d, cfg.d_model, bias=False)

    def forward(self, audio_mel: torch.Tensor, audio_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = audio_mel.shape
        x = self.frontend(audio_mel)
        pos = torch.arange(t, device=audio_mel.device).view(1, t).expand(b, t)
        for layer in self.layers:
            x = layer(x, pos)
        x = self.proj(x)
        x = x * audio_mask.unsqueeze(-1).to(x.dtype)
        return x, audio_mask
