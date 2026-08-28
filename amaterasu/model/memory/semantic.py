from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.constants import SEMANTIC_RING_SIZE


class SemanticMemory(nn.Module):
    """Q: d×d; K,V: d×512. Ring of keys/values is runtime state, not a table."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.ring_size = SEMANTIC_RING_SIZE
        self.k_sem = 8
        self.Wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.Wk = nn.Linear(cfg.d_model, 512, bias=False)
        self.Wv = nn.Linear(cfg.d_model, 512, bias=False)

    def retrieve(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        q = self.Wq(query)
        scores = torch.einsum("bd,bnd->bn", q, keys)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        k_use = min(self.k_sem, keys.shape[1])
        _, idx = scores.topk(k_use, dim=-1)
        gathered = torch.gather(values, 1, idx.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
        return gathered

    def write_ring(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        counts: torch.Tensor,
        cursor: torch.Tensor,
        event: torch.Tensor,
        do_write: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.Wk(event)
        v = self.Wv(event)
        full = valid.all(dim=-1)
        evict = torch.where(full, counts.argmin(dim=-1), cursor % self.ring_size)
        sel = do_write.bool()
        new_k, new_v = keys.clone(), values.clone()
        new_valid, new_counts, new_cursor = valid.clone(), counts.clone(), cursor.clone()
        if bool(sel.any()):
            rows = sel.nonzero(as_tuple=False).squeeze(-1)
            slot = evict[rows]
            new_k[rows, slot] = k[rows]
            new_v[rows, slot] = v[rows]
            new_valid[rows, slot] = True
            new_counts[rows, slot] = 0
            new_cursor[rows] = (slot + 1) % self.ring_size
        return new_k, new_v, new_valid, new_counts, new_cursor
