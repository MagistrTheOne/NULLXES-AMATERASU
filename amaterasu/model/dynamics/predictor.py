from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.transformer_block import PreNormBlock


class LatentDynamics(nn.Module):
    """8 layers d=3072 GQA 24/6/128 SwiGLU 8192 + three 4096→3072 IO projections."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.in_z = nn.Linear(cfg.d_model, cfg.dyn_d, bias=False)
        self.in_mem = nn.Linear(cfg.d_model, cfg.dyn_d, bias=False)
        self.in_cond = nn.Linear(cfg.d_model, cfg.dyn_d, bias=False)
        self.layers = nn.ModuleList(
            [
                PreNormBlock(
                    cfg.dyn_d,
                    cfg.dyn_heads,
                    cfg.dyn_kv,
                    cfg.dyn_d_head,
                    cfg.dyn_d_ff,
                    rope_theta=cfg.rope_theta,
                    eps=cfg.rms_eps,
                )
                for _ in range(cfg.dyn_layers)
            ]
        )

    def forward(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in_z(z)
        x = x + self.in_mem(memory).mean(dim=1, keepdim=True)
        x = x + self.in_cond(cond).mean(dim=1, keepdim=True)
        b, s, _ = x.shape
        pos = torch.arange(s, device=z.device).view(1, s).expand(b, s)
        for layer in self.layers:
            x = layer(x, pos)
        return x

    def predict_future(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        cond: torch.Tensor,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
    ) -> dict[int, torch.Tensor]:
        """JEPA-style latent futures. Caller stop-grads the target encoder."""
        pred = self.forward(z, memory, cond)
        return {k: pred for k in horizons}

    def encode_target(self, z_future: torch.Tensor) -> torch.Tensor:
        """Target encoder path: same in_z, no gradient if caller detaches z_future."""
        return self.in_z(z_future.detach())
