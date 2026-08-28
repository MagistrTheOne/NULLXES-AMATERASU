from __future__ import annotations

from dataclasses import dataclass, field

import torch

from amaterasu.tensors.modality import ModalityId


@dataclass
class AMATERASUSample:
    nces_feat: torch.Tensor
    nces_valid: torch.Tensor
    ecd_raw: torch.Tensor
    ecd_topo: torch.Tensor
    input_ids: torch.Tensor
    lang_mask: torch.Tensor
    video: torch.Tensor | None = None
    camera_valid_mask: torch.Tensor | None = None
    frame_times: torch.Tensor | None = None
    audio_mel: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None
    tactile: torch.Tensor | None = None
    tactile_valid: torch.Tensor | None = None
    nces_traj: torch.Tensor | None = None
    horizon_mask: torch.Tensor | None = None
    node_mask: torch.Tensor | None = None
    contact_idx: torch.Tensor | None = None
    contact_valid: torch.Tensor | None = None
    intent_label: torch.Tensor | None = None
    gate_label: torch.Tensor | None = None
    z_future: torch.Tensor | None = None
    episode_reset: bool = False
    null_instruction: bool = False
    license_ok_research: bool = True
    license_ok_commercial: bool = False
    source: str = ""
    token_time: torch.Tensor | None = None


@dataclass
class AMATERASUBatch:
    nces_feat: torch.Tensor
    nces_valid: torch.Tensor
    ecd_raw: torch.Tensor
    ecd_topo: torch.Tensor
    input_ids: torch.Tensor
    lang_mask: torch.Tensor
    video: torch.Tensor | None = None
    camera_valid_mask: torch.Tensor | None = None
    frame_times: torch.Tensor | None = None
    audio_mel: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None
    tactile: torch.Tensor | None = None
    tactile_valid: torch.Tensor | None = None
    nces_traj: torch.Tensor | None = None
    horizon_mask: torch.Tensor | None = None
    node_mask: torch.Tensor | None = None
    contact_idx: torch.Tensor | None = None
    contact_valid: torch.Tensor | None = None
    intent_label: torch.Tensor | None = None
    gate_label: torch.Tensor | None = None
    z_future: torch.Tensor | None = None
    episode_reset: torch.Tensor | None = None
    null_instruction: torch.Tensor | None = None
    flow_t: torch.Tensor | None = None
    modality_ids: torch.Tensor | None = None
    token_time: torch.Tensor | None = None
    source_ids: list[str] = field(default_factory=list)
