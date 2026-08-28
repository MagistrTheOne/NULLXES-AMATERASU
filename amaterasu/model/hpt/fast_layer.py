from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.attention.gqa import GQAAttention
from amaterasu.model.attention.mask import combine_additive_masks, sliding_window_mask
from amaterasu.model.ffn.swiglu import SwiGLU
from amaterasu.model.hpt.dispatch import apply_modality_ffn
from amaterasu.model.norms.rmsnorm import RMSNorm
from amaterasu.tensors.modality import ModalityId


def layer_attn_bias(
    layer_id: int,
    position_ids: torch.Tensor,
    temporal_bias: torch.Tensor | None,
    window: int,
) -> torch.Tensor | None:
    """3 sliding-window : 1 full, ANDed with block-structured temporal mask."""
    slide = None if (layer_id % 4 == 3) else sliding_window_mask(position_ids, window)
    return combine_additive_masks(temporal_bias, slide)


class HPTFastLayer(nn.Module):
    """L0–L11: shared GQA + Vision FFN + dense Physical FFN."""

    def __init__(self, cfg: Amaterasu32BConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.window = cfg.attn_window
        self.norm_attn = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = GQAAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.d_head,
            rope_theta=cfg.rope_theta,
            eps=cfg.rms_eps,
        )
        self.norm_vision = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_vision = SwiGLU(cfg.d_model, cfg.d_ff)
        self.norm_physical = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_physical = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        position_ids: torch.Tensor,
        temporal_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias = layer_attn_bias(self.layer_id, position_ids, temporal_bias, self.window)
        x = x + self.attn(self.norm_attn(x), position_ids, attn_bias=bias)
        x = x + apply_modality_ffn(
            x,
            modality_ids,
            {int(ModalityId.VISION): self.norm_vision, int(ModalityId.PHYSICAL): self.norm_physical},
            {int(ModalityId.VISION): self.ffn_vision, int(ModalityId.PHYSICAL): self.ffn_physical},
        )
        return x
