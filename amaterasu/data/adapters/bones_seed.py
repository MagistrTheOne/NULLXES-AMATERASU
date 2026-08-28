from __future__ import annotations

from pathlib import Path

import torch

from amaterasu.model.embodiment.ecd_pack import humanoid_ecd
from amaterasu.model.language.special_tokens import NULL_INSTRUCTION_ID, PAD_ID
from amaterasu.model.nces.convert import soma_joints_to_nces
from amaterasu.tensors.sample import AMATERASUSample


def record_to_sample(record: dict, root: Path | None = None) -> AMATERASUSample:
    """BONES-SEED / SOMA joints → NCES. Expects tensors already loaded by the caller."""
    pos = record["joint_pos"]
    rot = record["joint_rot6d"]
    valid = record["joint_valid"]
    feat, nvalid = soma_joints_to_nces(pos.unsqueeze(0), rot.unsqueeze(0), valid.unsqueeze(0))
    device = feat.device
    ecd_raw, ecd_topo = humanoid_ecd(1, device)
    ids = torch.tensor([NULL_INSTRUCTION_ID], dtype=torch.long, device=device)
    return AMATERASUSample(
        nces_feat=feat[0],
        nces_valid=nvalid[0],
        ecd_raw=ecd_raw[0],
        ecd_topo=ecd_topo[0],
        input_ids=ids,
        lang_mask=torch.ones(1, dtype=torch.bool, device=device),
        node_mask=nvalid[0],
        null_instruction=True,
        license_ok_research=True,
        license_ok_commercial=False,
        source="bones-seed",
        episode_reset=bool(record.get("episode_reset", False)),
    )
