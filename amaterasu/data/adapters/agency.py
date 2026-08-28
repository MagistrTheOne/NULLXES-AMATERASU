from __future__ import annotations

import torch

from amaterasu.model.embodiment.ecd_pack import humanoid_ecd
from amaterasu.model.language.special_tokens import NULL_INSTRUCTION_ID
from amaterasu.model.nces.convert import robot_obs_to_nces
from amaterasu.tensors.modality import IntentKind
from amaterasu.tensors.sample import AMATERASUSample


def record_to_sample(record: dict) -> AMATERASUSample:
    """Idle / OBSERVE / HOLD / WAIT / NULL-instruction / recovery labels."""
    feat, valid = robot_obs_to_nces(record["nces_feat"].unsqueeze(0), record["nces_valid"].unsqueeze(0))
    device = feat.device
    ecd_raw, ecd_topo = humanoid_ecd(1, device)
    kind = int(record.get("intent_kind", int(IntentKind.OBSERVE)))
    ids = record.get("input_ids")
    null = bool(record.get("null_instruction", True))
    if ids is None:
        ids = torch.tensor([NULL_INSTRUCTION_ID], dtype=torch.long, device=device)
    return AMATERASUSample(
        nces_feat=feat[0],
        nces_valid=valid[0],
        ecd_raw=ecd_raw[0],
        ecd_topo=ecd_topo[0],
        input_ids=ids,
        lang_mask=torch.ones(ids.shape[0], dtype=torch.bool, device=device),
        intent_label=torch.tensor([kind], dtype=torch.long, device=device),
        gate_label=record.get("gate_label"),
        nces_traj=record.get("nces_traj"),
        node_mask=valid[0],
        null_instruction=null,
        license_ok_research=True,
        license_ok_commercial=True,
        source="agency",
    )
