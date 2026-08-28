from __future__ import annotations

import torch

from amaterasu.constants import N_NODES_MAX
from amaterasu.model.nces.pack import pack_nodes, pad_nodes
from amaterasu.tensors.nces_schema import D_NCES_IN


def robot_obs_to_nces(
    packed: torch.Tensor,
    node_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity hook: already-canonical NCES features pass through."""
    if packed.shape[-1] != D_NCES_IN:
        raise ValueError(f"NCES features must be {D_NCES_IN}-d")
    return pad_nodes(packed, node_valid_mask.to(dtype=torch.bool), N_NODES_MAX)


def bimanual_ee_to_nces(
    left_xyz: torch.Tensor,
    left_rot6d: torch.Tensor,
    left_grip: torch.Tensor,
    right_xyz: torch.Tensor,
    right_rot6d: torch.Tensor,
    right_grip: torch.Tensor,
    root_xyz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HiFi-UMI-class 20-D bimanual xyz+rot6d+gripper → NCES nodes ROOT/L_HAND/R_HAND."""
    b = left_xyz.shape[0]
    device = left_xyz.device
    dtype = left_xyz.dtype
    n = 3
    pos = torch.zeros(b, n, 3, device=device, dtype=dtype)
    rot = torch.zeros(b, n, 6, device=device, dtype=dtype)
    lin = torch.zeros(b, n, 3, device=device, dtype=dtype)
    ang = torch.zeros(b, n, 3, device=device, dtype=dtype)
    valid = torch.ones(b, n, dtype=torch.bool, device=device)
    grasp = torch.zeros(b, n, 2, device=device, dtype=dtype)
    if root_xyz is not None:
        pos[:, 0] = root_xyz
    else:
        pos[:, 0] = 0.5 * (left_xyz + right_xyz)
    rot[:, 0, 0] = 1.0
    rot[:, 0, 4] = 1.0
    pos[:, 1] = left_xyz
    rot[:, 1] = left_rot6d
    grasp[:, 1, 0] = left_grip.reshape(b)
    pos[:, 2] = right_xyz
    rot[:, 2] = right_rot6d
    grasp[:, 2, 0] = right_grip.reshape(b)
    feat, valid = pack_nodes(pos, rot, lin, ang, valid, grasp=grasp)
    return pad_nodes(feat, valid, N_NODES_MAX)


def soma_joints_to_nces(
    joint_pos: torch.Tensor,
    joint_rot6d: torch.Tensor,
    joint_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BONES-SEED / SOMA joints [B,J,3] + [B,J,6] mapped onto the first J NCES nodes."""
    lin = torch.zeros_like(joint_pos)
    ang = torch.zeros_like(joint_pos)
    feat, valid = pack_nodes(joint_pos, joint_rot6d, lin, ang, joint_valid)
    return pad_nodes(feat, valid, N_NODES_MAX)
