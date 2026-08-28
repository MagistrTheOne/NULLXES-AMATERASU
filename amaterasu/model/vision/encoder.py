from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.transformer_block import PreNormBlock
from amaterasu.model.vision.tubelet import tubelet_stem


class VisionEncoder(nn.Module):
    """36-layer tubelet ViT at d=2048, project to d_model. No learned positional table."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        tt, th, tw, c = cfg.tubelet
        self.tubelet_t = tt
        self.tubelet_h = th
        self.tubelet_w = tw
        self.tubelet = tubelet_stem(c, cfg.ve_d, tt, th, tw)
        self.layers = nn.ModuleList(
            [
                PreNormBlock(
                    cfg.ve_d,
                    cfg.ve_heads,
                    cfg.ve_kv,
                    cfg.ve_d_head,
                    cfg.ve_d_ff,
                    rope_theta=cfg.rope_theta,
                    eps=cfg.rms_eps,
                )
                for _ in range(cfg.ve_layers)
            ]
        )
        self.proj = nn.Linear(cfg.ve_d, cfg.d_model, bias=False)

    def forward(
        self,
        video: torch.Tensor,
        camera_valid_mask: torch.Tensor,
        frame_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """video [B,N,T,C,H,W], camera_valid_mask [B,N], frame_times [B,N,T].

        Returns hpt_vision, vision_mask, vision_time.
        """
        b, n, t, c, h, w = video.shape
        if t % self.tubelet_t != 0 or h % self.tubelet_h != 0 or w % self.tubelet_w != 0:
            raise ValueError("video T,H,W must be divisible by tubelet (2,14,14)")
        x = video.reshape(b * n, c, t, h, w)
        x = self.tubelet(x)
        _, d, tt, hh, ww = x.shape
        tok = x.flatten(2).transpose(1, 2)
        s = tok.shape[1]
        pos = torch.arange(s, device=video.device).view(1, s).expand(b * n, s)
        for layer in self.layers:
            tok = layer(tok, pos)
        tok = self.proj(tok)
        tok = tok.view(b, n, s, tok.shape[-1])
        valid = camera_valid_mask.to(dtype=torch.bool).unsqueeze(-1)
        tok = tok * valid.unsqueeze(-1).to(tok.dtype)
        hpt = tok.reshape(b, n * s, tok.shape[-1])
        vision_mask = valid.expand(b, n, s).reshape(b, n * s)
        later = frame_times[:, :, self.tubelet_t - 1 :: self.tubelet_t]
        if later.shape[-1] != tt:
            later = frame_times.reshape(b, n, tt, self.tubelet_t).mean(dim=-1)
        vision_time = later.unsqueeze(-1).expand(b, n, tt, hh * ww).reshape(b, n * s)
        vision_time = torch.where(vision_mask, vision_time, vision_time.new_zeros(vision_time.shape))
        return hpt, vision_mask, vision_time
