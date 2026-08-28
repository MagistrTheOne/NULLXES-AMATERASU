from __future__ import annotations

import torch


def block_structured_temporal_mask(
    tick_ids: torch.Tensor,
    key_valid: torch.Tensor,
) -> torch.Tensor:
    """Additive mask [B, 1, S, S] for SDPA.

    Query i attends to key j iff tick[j] <= tick[i] and key j is valid.
    Same-tick tokens fuse bidirectionally. Future observation leak is blocked.
    """
    q = tick_ids.unsqueeze(-1)
    k = tick_ids.unsqueeze(-2)
    visible = (k <= q) & key_valid.unsqueeze(-2)
    add = torch.zeros(visible.shape, dtype=torch.float32, device=tick_ids.device)
    add = add.masked_fill(~visible, torch.finfo(torch.float32).min)
    return add.unsqueeze(1)


def sliding_window_mask(position_ids: torch.Tensor, window: int) -> torch.Tensor:
    """Additive mask [B, 1, S, S]: keep keys within `window` of the query position."""
    q = position_ids.unsqueeze(-1)
    k = position_ids.unsqueeze(-2)
    visible = (q - k) < window
    add = torch.zeros(visible.shape, dtype=torch.float32, device=position_ids.device)
    add = add.masked_fill(~visible, torch.finfo(torch.float32).min)
    return add.unsqueeze(1)


def combine_additive_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    acc: torch.Tensor | None = None
    for m in masks:
        if m is None:
            continue
        acc = m if acc is None else torch.minimum(acc, m)
    return acc
