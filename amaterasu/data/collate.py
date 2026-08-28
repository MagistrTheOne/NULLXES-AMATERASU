from __future__ import annotations

import torch

from amaterasu.tensors.sample import AMATERASUBatch, AMATERASUSample


def _stack_optional(xs: list[torch.Tensor | None], name: str) -> torch.Tensor | None:
    if all(x is None for x in xs):
        return None
    if any(x is None for x in xs):
        raise ValueError(f"batch field {name} is mixed None/tensor")
    return torch.stack([x for x in xs if x is not None], dim=0)


def collate_samples(samples: list[AMATERASUSample]) -> AMATERASUBatch:
    if not samples:
        raise ValueError("empty batch")
    return AMATERASUBatch(
        nces_feat=torch.stack([s.nces_feat for s in samples], 0),
        nces_valid=torch.stack([s.nces_valid for s in samples], 0),
        ecd_raw=torch.stack([s.ecd_raw for s in samples], 0),
        ecd_topo=torch.stack([s.ecd_topo for s in samples], 0),
        input_ids=torch.stack([s.input_ids for s in samples], 0),
        lang_mask=torch.stack([s.lang_mask for s in samples], 0),
        video=_stack_optional([s.video for s in samples], "video"),
        camera_valid_mask=_stack_optional([s.camera_valid_mask for s in samples], "camera_valid_mask"),
        frame_times=_stack_optional([s.frame_times for s in samples], "frame_times"),
        audio_mel=_stack_optional([s.audio_mel for s in samples], "audio_mel"),
        audio_mask=_stack_optional([s.audio_mask for s in samples], "audio_mask"),
        tactile=_stack_optional([s.tactile for s in samples], "tactile"),
        tactile_valid=_stack_optional([s.tactile_valid for s in samples], "tactile_valid"),
        nces_traj=_stack_optional([s.nces_traj for s in samples], "nces_traj"),
        horizon_mask=_stack_optional([s.horizon_mask for s in samples], "horizon_mask"),
        node_mask=_stack_optional([s.node_mask for s in samples], "node_mask"),
        contact_idx=_stack_optional([s.contact_idx for s in samples], "contact_idx"),
        contact_valid=_stack_optional([s.contact_valid for s in samples], "contact_valid"),
        intent_label=_stack_optional([s.intent_label for s in samples], "intent_label"),
        gate_label=_stack_optional([s.gate_label for s in samples], "gate_label"),
        z_future=_stack_optional([s.z_future for s in samples], "z_future"),
        episode_reset=torch.tensor([s.episode_reset for s in samples], dtype=torch.bool),
        null_instruction=torch.tensor([s.null_instruction for s in samples], dtype=torch.bool),
        source_ids=[s.source for s in samples],
    )
