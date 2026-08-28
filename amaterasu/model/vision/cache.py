from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VisualCache:
    """Cached visual tokens between FAST_SENSOR_REFRESH ticks. Not parameters."""

    hpt_vision: torch.Tensor
    vision_mask: torch.Tensor
    vision_time: torch.Tensor
    cache_id: torch.Tensor

    def replace(
        self,
        hpt_vision: torch.Tensor,
        vision_mask: torch.Tensor,
        vision_time: torch.Tensor,
    ) -> "VisualCache":
        return VisualCache(
            hpt_vision=hpt_vision,
            vision_mask=vision_mask,
            vision_time=vision_time,
            cache_id=self.cache_id + 1,
        )


def empty_visual_cache(batch: int, n_vis: int, d_model: int, device: torch.device) -> VisualCache:
    return VisualCache(
        hpt_vision=torch.zeros(batch, n_vis, d_model, device=device),
        vision_mask=torch.zeros(batch, n_vis, dtype=torch.bool, device=device),
        vision_time=torch.zeros(batch, n_vis, device=device),
        cache_id=torch.zeros(batch, dtype=torch.long, device=device),
    )
