from __future__ import annotations

import torch

from amaterasu.model.embodiment.ecd_pack import pack_ecd, empty_topo
from amaterasu.model.language.special_tokens import NULL_INSTRUCTION_ID
from amaterasu.model.nces.convert import bimanual_ee_to_nces
from amaterasu.tensors.sample import AMATERASUSample


def record_to_sample(record: dict) -> AMATERASUSample:
    """HiFi-UMI-2K 20-D bimanual xyz+rot6d+gripper → NCES. Video tensors optional."""
    feat, valid = bimanual_ee_to_nces(
        record["left_xyz"].unsqueeze(0),
        record["left_rot6d"].unsqueeze(0),
        record["left_grip"].view(1),
        record["right_xyz"].unsqueeze(0),
        record["right_rot6d"].unsqueeze(0),
        record["right_grip"].view(1),
    )
    device = feat.device
    topo = torch.zeros(1, 16, device=device)
    topo[:, 1] = 1.0
    effectors = torch.zeros(1, 16, device=device)
    effectors[:, :2] = 1.0
    ecd_raw = pack_ecd(
        topo,
        effectors,
        torch.zeros(1, 16, device=device),
        torch.zeros(1, 8, device=device),
        torch.ones(1, 8, device=device),
        torch.ones(1, 16, device=device),
        torch.zeros(1, 8, device=device),
        torch.zeros(1, 8, device=device),
        torch.zeros(1, 16, device=device),
    )
    ids = torch.tensor([NULL_INSTRUCTION_ID], dtype=torch.long, device=device)
    video = record.get("video")
    cam_mask = record.get("camera_valid_mask")
    times = record.get("frame_times")
    traj = record.get("nces_traj")
    return AMATERASUSample(
        nces_feat=feat[0],
        nces_valid=valid[0],
        ecd_raw=ecd_raw[0],
        ecd_topo=empty_topo(1, device)[0],
        input_ids=ids,
        lang_mask=torch.ones(1, dtype=torch.bool, device=device),
        video=video,
        camera_valid_mask=cam_mask,
        frame_times=times,
        nces_traj=traj,
        node_mask=valid[0],
        horizon_mask=record.get("horizon_mask"),
        null_instruction=True,
        license_ok_research=True,
        license_ok_commercial=True,
        source="hifi-umi-2k",
    )
