from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.constants import EPISODIC_RING_SIZE
from amaterasu.model.transformer_block import PreNormBlock


class EpisodicMemory(nn.Module):
    """2 VE-width compressor layers + 3 * 2048 * 4096 projections. Ring is runtime state."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.ring_size = EPISODIC_RING_SIZE
        self.in_proj = nn.Linear(cfg.d_model, cfg.ve_d, bias=False)
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
                for _ in range(2)
            ]
        )
        self.out_proj = nn.Linear(cfg.ve_d, cfg.d_model, bias=False)
        self.write_proj = nn.Linear(cfg.ve_d, cfg.d_model, bias=False)

    def compress(self, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = events.shape
        x = self.in_proj(events)
        pos = torch.arange(s, device=events.device).view(1, s).expand(b, s)
        for layer in self.layers:
            x = layer(x, pos)
        return self.out_proj(x), self.write_proj(x.mean(dim=1, keepdim=True))

    def write_ring(
        self,
        ring: torch.Tensor,
        valid: torch.Tensor,
        cursor: torch.Tensor,
        compressed: torch.Tensor,
        do_write: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ring [B, R, d]; overflow evicts oldest. Differentiable in `compressed` for written rows."""
        b = ring.shape[0]
        idx = cursor % self.ring_size
        sel = do_write.bool()
        new_ring = ring.clone()
        new_valid = valid.clone()
        new_cursor = cursor.clone()
        if bool(sel.any()):
            rows = sel.nonzero(as_tuple=False).squeeze(-1)
            new_ring[rows, idx[rows]] = compressed[rows, 0]
            new_valid[rows, idx[rows]] = True
            new_cursor[rows] = (idx[rows] + 1) % self.ring_size
        return new_ring, new_valid, new_cursor

    def read(self, ring: torch.Tensor, valid: torch.Tensor, query: torch.Tensor, k: int = 8) -> torch.Tensor:
        scores = torch.einsum("bd,brd->br", query, ring)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        k_use = min(k, ring.shape[1])
        _, idx = scores.topk(k_use, dim=-1)
        gather = idx.unsqueeze(-1).expand(-1, -1, ring.shape[-1])
        return torch.gather(ring, 1, gather)
