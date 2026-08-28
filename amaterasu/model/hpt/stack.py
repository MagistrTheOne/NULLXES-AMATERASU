from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.hpt.fast_layer import HPTFastLayer
from amaterasu.model.hpt.slow_layer import HPTSlowDenseLayer, HPTSlowMoELayer


class HPTStack(nn.Module):
    """40-layer Heterogeneous Physical Transformer."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        fast = [HPTFastLayer(cfg, i) for i in range(cfg.n_fast_layers)]
        slow_dense = [
            HPTSlowDenseLayer(cfg, cfg.n_fast_layers + i) for i in range(cfg.n_slow_dense_physical)
        ]
        slow_moe = [
            HPTSlowMoELayer(cfg, cfg.n_fast_layers + cfg.n_slow_dense_physical + i)
            for i in range(cfg.n_moe_layers)
        ]
        self.fast_layers = nn.ModuleList(fast)
        self.slow_dense_layers = nn.ModuleList(slow_dense)
        self.slow_moe_layers = nn.ModuleList(slow_moe)

    def forward_fast(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        position_ids: torch.Tensor,
        temporal_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.fast_layers:
            x = layer(x, modality_ids, position_ids, temporal_bias)
        return x

    def forward_slow(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        position_ids: torch.Tensor,
        temporal_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.slow_dense_layers:
            x = layer(x, modality_ids, position_ids, temporal_bias)
        for layer in self.slow_moe_layers:
            x = layer(x, modality_ids, position_ids, temporal_bias)
        return x

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        position_ids: torch.Tensor,
        temporal_bias: torch.Tensor | None = None,
        run_slow: bool = True,
    ) -> torch.Tensor:
        x = self.forward_fast(x, modality_ids, position_ids, temporal_bias)
        if run_slow:
            x = self.forward_slow(x, modality_ids, position_ids, temporal_bias)
        return x
