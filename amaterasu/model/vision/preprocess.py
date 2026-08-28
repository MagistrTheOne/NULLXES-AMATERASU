from __future__ import annotations

import torch
import torch.nn.functional as F

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.constants import N_CAM_MAX, T_CLIP, TRAIN_H, TRAIN_W


def _center_crop_spatial(video: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """video [B,N,T,C,H,W] → crop H,W to th,tw (or leave if already smaller)."""
    _, _, _, _, h, w = video.shape
    if h == th and w == tw:
        return video
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    out_h = min(th, h)
    out_w = min(tw, w)
    return video[:, :, :, :, top : top + out_h, left : left + out_w]


def _resize_spatial(video: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    b, n, t, c, h, w = video.shape
    if h == th and w == tw:
        return video
    x = video.reshape(b * n * t, c, h, w)
    x = F.interpolate(x.float(), size=(th, tw), mode="bilinear", align_corners=False)
    return x.to(video.dtype).reshape(b, n, t, c, th, tw)


def _align_time(video: torch.Tensor, times: torch.Tensor, t_clip: int, tubelet_t: int) -> tuple[torch.Tensor, torch.Tensor]:
    t = video.shape[2]
    need = t_clip if t_clip % tubelet_t == 0 else t_clip - (t_clip % tubelet_t)
    if t >= need:
        return video[:, :, :need], times[:, :, :need]
    pad_t = need - t
    video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_t))
    last = times[:, :, -1:]
    extra = last.expand(-1, -1, pad_t)
    return video, torch.cat([times, extra], dim=2)


def ingest_video(
    video: torch.Tensor,
    camera_valid_mask: torch.Tensor,
    frame_times: torch.Tensor,
    cfg: Amaterasu32BConfig | None = None,
    crop: bool = True,
    resize_to_train: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native-resolution ingest → optional crop/resize. Does not duplicate cameras.

    video [B,N,T,C,H,W], mask [B,N], times [B,N,T] in seconds (async per camera).
    N may be < N_CAM_MAX; padded cameras stay invalid.
    """
    cfg = cfg or Amaterasu32BConfig()
    if video.ndim != 6:
        raise ValueError("video must be [B,N,T,C,H,W]")
    b, n, t, c, h, w = video.shape
    if c != 3:
        raise ValueError("video C must be 3")
    if n > N_CAM_MAX:
        video = video[:, :N_CAM_MAX]
        camera_valid_mask = camera_valid_mask[:, :N_CAM_MAX]
        frame_times = frame_times[:, :N_CAM_MAX]
        n = N_CAM_MAX
    if n < N_CAM_MAX:
        pad_n = N_CAM_MAX - n
        video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_n))
        camera_valid_mask = F.pad(camera_valid_mask, (0, pad_n))
        frame_times = F.pad(frame_times, (0, 0, 0, pad_n))
    if crop:
        th = (h // cfg.tubelet[1]) * cfg.tubelet[1]
        tw = (w // cfg.tubelet[2]) * cfg.tubelet[2]
        if th < cfg.tubelet[1] or tw < cfg.tubelet[2]:
            raise ValueError("spatial size smaller than tubelet")
        video = _center_crop_spatial(video, th, tw)
    if resize_to_train:
        video = _resize_spatial(video, cfg.train_h, cfg.train_w)
    video, frame_times = _align_time(video, frame_times, cfg.t_clip, cfg.tubelet[0])
    camera_valid_mask = camera_valid_mask.to(dtype=torch.bool)
    video = video * camera_valid_mask.view(video.shape[0], video.shape[1], 1, 1, 1, 1).to(video.dtype)
    frame_times = torch.where(
        camera_valid_mask.unsqueeze(-1),
        frame_times,
        frame_times.new_zeros(frame_times.shape),
    )
    return video, camera_valid_mask, frame_times


def train_geometry() -> tuple[int, int, int, int]:
    return N_CAM_MAX, T_CLIP, TRAIN_H, TRAIN_W
