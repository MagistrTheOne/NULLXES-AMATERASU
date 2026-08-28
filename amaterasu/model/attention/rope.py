from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rotary_from_pos(x: torch.Tensor, position_ids: torch.Tensor, theta: float) -> torch.Tensor:
    """x: [B, S, H, Dh] even Dh. position_ids: [B, S]."""
    dh = x.shape[-1]
    device = x.device
    freq = 1.0 / (theta ** (torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh))
    pos = position_ids.to(dtype=torch.float32).unsqueeze(-1)
    ang = pos * freq
    cos = torch.cos(ang).repeat_interleave(2, dim=-1).unsqueeze(2)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1).unsqueeze(2)
    x32 = x.float()
    return (x32 * cos + _rotate_half(x32) * sin).to(x.dtype)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
    k_position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_pos = position_ids if k_position_ids is None else k_position_ids
    return rotary_from_pos(q, position_ids, theta), rotary_from_pos(k, k_pos, theta)
