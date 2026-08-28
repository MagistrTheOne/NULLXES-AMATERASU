from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.agency.aux_heads import AuxHeads
from amaterasu.model.agency.gate import ConstraintGate
from amaterasu.model.agency.q_theta import QTheta
from amaterasu.model.transformer_block import PreNormBlock


class GCIS(nn.Module):
    """4-layer decoder-width stack over agency tokens + 11 intent queries."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.agency_tokens = nn.Parameter(torch.zeros(cfg.eac_agency_tokens, cfg.d_model))
        self.intent_queries = nn.Parameter(torch.zeros(cfg.eac_queries, cfg.d_model))
        self.layers = nn.ModuleList(
            [
                PreNormBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.d_head,
                    cfg.d_ff,
                    rope_theta=cfg.rope_theta,
                    eps=cfg.rms_eps,
                )
                for _ in range(cfg.eac_gcis_layers)
            ]
        )
        self.q_theta = QTheta(cfg)
        self.gate = ConstraintGate(cfg)
        self.aux = AuxHeads(cfg)

    def forward(
        self,
        context: torch.Tensor,
        z_pool: torch.Tensor,
        g_pool: torch.Tensor,
        context_pos: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, s, _ = context.shape
        agency = self.agency_tokens.unsqueeze(0).expand(b, -1, -1)
        queries = self.intent_queries.unsqueeze(0).expand(b, -1, -1)
        tokens = torch.cat([agency, queries, context], dim=1)
        n_a = agency.shape[1]
        n_q = queries.shape[1]
        pos_a = torch.arange(n_a, device=context.device).view(1, n_a).expand(b, n_a)
        pos_q = torch.arange(n_q, device=context.device).view(1, n_q).expand(b, n_q) + n_a
        if context_pos is None:
            pos_c = torch.arange(s, device=context.device).view(1, s).expand(b, s) + n_a + n_q
        else:
            pos_c = context_pos + n_a + n_q
        pos = torch.cat([pos_a, pos_q, pos_c], dim=1)
        x = tokens
        for layer in self.layers:
            x = layer(x, pos)
        agency_out = x[:, :n_a]
        intent = x[:, n_a : n_a + n_q]
        a_pool = agency_out.mean(dim=1)
        aux = self.aux(a_pool)
        side32 = torch.zeros(b, 32, device=context.device, dtype=torch.float32)
        side32[:, : aux.shape[-1]] = aux.float()
        scores = self.q_theta(intent, z_pool, a_pool, g_pool)
        gate_logits = self.gate(intent, side32)
        kinds = torch.arange(n_q, device=context.device, dtype=torch.int8).view(1, n_q).expand(b, n_q)
        return {
            "agency_tokens": agency_out,
            "intent_latents": intent,
            "intent_scores": scores.float(),
            "gate_logits": gate_logits.float(),
            "aux_heads": aux.float(),
            "intent_kind": kinds,
            "intent_mask": torch.ones(b, n_q, dtype=torch.bool, device=context.device),
        }
