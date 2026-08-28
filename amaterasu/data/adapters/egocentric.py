from __future__ import annotations

import torch

from amaterasu.model.embodiment.ecd_pack import humanoid_ecd
from amaterasu.model.language.special_tokens import NULL_INSTRUCTION_ID
from amaterasu.model.nces.pack import empty_nces
from amaterasu.tensors.sample import AMATERASUSample


def record_to_sample(record: dict) -> AMATERASUSample:
    """Egocentric video observation. NCES may be empty/invalid if no proprioception."""
    video = record["video"]
    device = video.device
    feat = empty_nces(1, record.get("n_nodes", 64), device, video.dtype)[0]
    valid = torch.zeros(feat.shape[0], dtype=torch.bool, device=device)
    ecd_raw, ecd_topo = humanoid_ecd(1, device)
    ids = record.get("input_ids", torch.tensor([NULL_INSTRUCTION_ID], dtype=torch.long, device=device))
    src = str(record.get("source", "ego4d"))
    return AMATERASUSample(
        nces_feat=feat,
        nces_valid=valid,
        ecd_raw=ecd_raw[0],
        ecd_topo=ecd_topo[0],
        input_ids=ids,
        lang_mask=torch.ones(ids.shape[0], dtype=torch.bool, device=device),
        video=video,
        camera_valid_mask=record["camera_valid_mask"],
        frame_times=record["frame_times"],
        null_instruction=True,
        license_ok_research=True,
        license_ok_commercial=False,
        source=src,
    )
