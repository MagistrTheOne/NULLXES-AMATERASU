from __future__ import annotations

import torch

from amaterasu.model.embodiment.ecd_pack import pack_ecd, empty_topo
from amaterasu.model.language.special_tokens import NULL_INSTRUCTION_ID
from amaterasu.model.nces.convert import robot_obs_to_nces
from amaterasu.tensors.sample import AMATERASUSample


def record_to_sample(record: dict) -> AMATERASUSample:
    feat, valid = robot_obs_to_nces(record["nces_feat"].unsqueeze(0), record["nces_valid"].unsqueeze(0))
    device = feat.device
    ecd_raw = record.get("ecd_raw")
    if ecd_raw is None:
        ecd_raw = pack_ecd(
            torch.zeros(1, 16, device=device),
            torch.zeros(1, 16, device=device),
            torch.zeros(1, 16, device=device),
            torch.zeros(1, 8, device=device),
            torch.zeros(1, 8, device=device),
            torch.zeros(1, 16, device=device),
            torch.zeros(1, 8, device=device),
            torch.zeros(1, 8, device=device),
            torch.zeros(1, 16, device=device),
        )[0]
    else:
        ecd_raw = ecd_raw if ecd_raw.ndim == 1 else ecd_raw[0]
    ids = record.get("input_ids", torch.tensor([NULL_INSTRUCTION_ID], dtype=torch.long, device=device))
    src = str(record.get("source", "droid"))
    commercial = src.lower() in {"droid", "bridge", "oxe"}
    return AMATERASUSample(
        nces_feat=feat[0],
        nces_valid=valid[0],
        ecd_raw=ecd_raw,
        ecd_topo=record.get("ecd_topo", empty_topo(1, device)[0]),
        input_ids=ids,
        lang_mask=record.get("lang_mask", torch.ones(ids.shape[0], dtype=torch.bool, device=device)),
        nces_traj=record.get("nces_traj"),
        horizon_mask=record.get("horizon_mask"),
        node_mask=valid[0],
        null_instruction=bool(record.get("null_instruction", True)),
        license_ok_research=True,
        license_ok_commercial=commercial,
        source=src,
    )
