from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.attention.gqa import GQAAttention
from amaterasu.model.experts.physical_moe import PhysicalMoE
from amaterasu.model.ffn.swiglu import SwiGLU
from amaterasu.model.hpt.dispatch import apply_modality_ffn, physical_token_mask
from amaterasu.model.hpt.fast_layer import layer_attn_bias
from amaterasu.model.norms.rmsnorm import RMSNorm
from amaterasu.tensors.modality import ModalityId


class HPTSlowDenseLayer(nn.Module):
    """L12–L19: four dense modality FFNs."""

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
        self.norm_language = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_language = SwiGLU(cfg.d_model, cfg.d_ff)
        self.norm_physical = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_physical = SwiGLU(cfg.d_model, cfg.d_ff)
        self.norm_agency = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_agency = SwiGLU(cfg.d_model, cfg.d_ff)

    def _ffn_maps(self) -> tuple[dict[int, nn.Module], dict[int, nn.Module]]:
        norms = {
            int(ModalityId.VISION): self.norm_vision,
            int(ModalityId.LANGUAGE): self.norm_language,
            int(ModalityId.PHYSICAL): self.norm_physical,
            int(ModalityId.AGENCY): self.norm_agency,
        }
        ffns = {
            int(ModalityId.VISION): self.ffn_vision,
            int(ModalityId.LANGUAGE): self.ffn_language,
            int(ModalityId.PHYSICAL): self.ffn_physical,
            int(ModalityId.AGENCY): self.ffn_agency,
        }
        return norms, ffns

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        position_ids: torch.Tensor,
        temporal_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias = layer_attn_bias(self.layer_id, position_ids, temporal_bias, self.window)
        x = x + self.attn(self.norm_attn(x), position_ids, attn_bias=bias)
        norms, ffns = self._ffn_maps()
        return x + apply_modality_ffn(x, modality_ids, norms, ffns)


class HPTSlowMoELayer(nn.Module):
    """L20–L39: Vision/Language/Agency dense FFNs + sparse Physical MoE."""

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
        self.norm_language = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_language = SwiGLU(cfg.d_model, cfg.d_ff)
        self.norm_physical = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.moe = PhysicalMoE(cfg.d_model, cfg.d_ff_expert, cfg.n_routed_experts, cfg.moe_topk)
        self.norm_agency = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_agency = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        position_ids: torch.Tensor,
        temporal_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bias = layer_attn_bias(self.layer_id, position_ids, temporal_bias, self.window)
        x = x + self.attn(self.norm_attn(x), position_ids, attn_bias=bias)
        dense_norms = {
            int(ModalityId.VISION): self.norm_vision,
            int(ModalityId.LANGUAGE): self.norm_language,
            int(ModalityId.AGENCY): self.norm_agency,
        }
        dense_ffns = {
            int(ModalityId.VISION): self.ffn_vision,
            int(ModalityId.LANGUAGE): self.ffn_language,
            int(ModalityId.AGENCY): self.ffn_agency,
        }
        x = x + apply_modality_ffn(x, modality_ids, dense_norms, dense_ffns)
        phys = physical_token_mask(modality_ids)
        phys_in = self.norm_physical(x)
        x = x + self.moe(phys_in, phys)
        return x
