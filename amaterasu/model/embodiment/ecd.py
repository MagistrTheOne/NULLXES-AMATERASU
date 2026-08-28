from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.transformer_block import PreNormBlock


class ECDEncoder(nn.Module):
    """Capability MLP 128→1024→4096 plus 3-layer d=512 topology encoder → 4096."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.cap_fc1 = nn.Linear(cfg.ecd_in, cfg.ecd_hidden, bias=False)
        self.cap_fc2 = nn.Linear(cfg.ecd_hidden, cfg.d_model, bias=False)
        self.topo_in = nn.Linear(cfg.ecd_topo * cfg.ecd_topo, cfg.ecd_d, bias=False)
        self.layers = nn.ModuleList(
            [
                PreNormBlock(
                    cfg.ecd_d,
                    n_heads=8,
                    n_kv_heads=8,
                    d_head=64,
                    d_ff=cfg.ecd_d_ff,
                    rope_theta=cfg.rope_theta,
                    eps=cfg.rms_eps,
                    use_qk_norm=False,
                )
                for _ in range(cfg.ecd_layers)
            ]
        )
        self.topo_out = nn.Linear(cfg.ecd_d, cfg.d_model, bias=False)

    def forward(self, ecd_raw: torch.Tensor, ecd_topo: torch.Tensor) -> torch.Tensor:
        cap = self.cap_fc2(torch.nn.functional.silu(self.cap_fc1(ecd_raw)))
        b = ecd_raw.shape[0]
        topo = self.topo_in(ecd_topo.reshape(b, -1)).unsqueeze(1)
        pos = torch.zeros(b, 1, dtype=torch.long, device=ecd_raw.device)
        for layer in self.layers:
            topo = layer(topo, pos)
        topo = self.topo_out(topo)
        cap = cap.unsqueeze(1)
        return torch.cat([cap, topo], dim=1)
