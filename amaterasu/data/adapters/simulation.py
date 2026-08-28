from __future__ import annotations

import torch

from amaterasu.model.embodiment.ecd_pack import humanoid_ecd
from amaterasu.model.language.special_tokens import NULL_INSTRUCTION_ID
from amaterasu.model.nces.convert import robot_obs_to_nces
from amaterasu.tensors.sample import AMATERASUSample


def record_to_sample(record: dict) -> AMATERASUSample:
    """Sim counterfactual / dynamics branch. Factual Z_future required; do not treat logs as counterfactuals."""
    feat, valid = robot_obs_to_nces(record["nces_feat"].unsqueeze(0), record["nces_valid"].unsqueeze(0))
    device = feat.device
    ecd_raw, ecd_topo = humanoid_ecd(1, device)
    ids = torch.tensor([NULL_INSTRUCTION_ID], dtype=torch.long, device=device)
    return AMATERASUSample(
        nces_feat=feat[0],
        nces_valid=valid[0],
        ecd_raw=ecd_raw[0],
        ecd_topo=ecd_topo[0],
        input_ids=ids,
        lang_mask=torch.ones(1, dtype=torch.bool, device=device),
        nces_traj=record.get("nces_traj"),
        horizon_mask=record.get("horizon_mask"),
        node_mask=valid[0],
        z_future=record.get("z_future"),
        intent_label=record.get("intent_label"),
        null_instruction=True,
        license_ok_research=True,
        license_ok_commercial=True,
        source="simulation",
    )
