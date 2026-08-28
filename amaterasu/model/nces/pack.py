from __future__ import annotations

import torch

from amaterasu.constants import N_NODES_MAX
from amaterasu.tensors.nces_schema import (
    ANG_VEL,
    CONTACT_BIN,
    D_NCES_IN,
    FRAME_FLAGS,
    GRASP,
    GRAVITY,
    LIN_VEL,
    MOMENTUM,
    NODE_VALID_IN_FEAT,
    POS,
    ROT6D,
    SUPPORT,
    WRENCH,
)


def empty_nces(batch: int, n_nodes: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(batch, n_nodes, D_NCES_IN, device=device, dtype=dtype)


def pack_nodes(
    position: torch.Tensor,
    rot6d: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    node_valid: torch.Tensor,
    grasp: torch.Tensor | None = None,
    contact_binary: torch.Tensor | None = None,
    wrench: torch.Tensor | None = None,
    gravity: torch.Tensor | None = None,
    support: torch.Tensor | None = None,
    momentum: torch.Tensor | None = None,
    frame_flags: torch.Tensor | None = None,
    topology_onehot: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack structured fields into [B, N, 128]. Validity is a separate mask, not implied by zeros."""
    b, n, _ = position.shape
    feat = position.new_zeros(b, n, D_NCES_IN)
    feat[..., POS] = position
    feat[..., ROT6D] = rot6d
    feat[..., LIN_VEL] = lin_vel
    feat[..., ANG_VEL] = ang_vel
    feat[..., NODE_VALID_IN_FEAT] = node_valid.to(feat.dtype).unsqueeze(-1)
    if grasp is not None:
        feat[..., GRASP] = grasp
    if contact_binary is not None:
        feat[..., CONTACT_BIN] = contact_binary.to(feat.dtype).unsqueeze(-1)
    if wrench is not None:
        feat[..., WRENCH] = wrench
    if gravity is not None:
        feat[..., GRAVITY] = gravity
    if support is not None:
        feat[..., SUPPORT] = support
    if momentum is not None:
        feat[..., MOMENTUM] = momentum
    if frame_flags is not None:
        feat[..., FRAME_FLAGS] = frame_flags
    if topology_onehot is not None:
        k = min(topology_onehot.shape[-1], 96)
        feat[..., 32 : 32 + k] = topology_onehot[..., :k]
    feat = feat * node_valid.unsqueeze(-1).to(feat.dtype)
    return feat, node_valid.to(dtype=torch.bool)


def pad_nodes(feat: torch.Tensor, valid: torch.Tensor, n_max: int = N_NODES_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, d = feat.shape
    if n > n_max:
        return feat[:, :n_max], valid[:, :n_max]
    if n == n_max:
        return feat, valid
    pad = feat.new_zeros(b, n_max - n, d)
    vpad = valid.new_zeros(b, n_max - n)
    return torch.cat([feat, pad], dim=1), torch.cat([valid, vpad], dim=1)


def topology_mask(n_nodes: int, edges: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Undirected adjacency [N,N] from edge index [E,2]. Diagonal is True for valid self."""
    adj = torch.eye(n_nodes, device=device, dtype=torch.bool)
    if edges.numel() == 0:
        return adj
    i, j = edges[:, 0].long(), edges[:, 1].long()
    adj[i, j] = True
    adj[j, i] = True
    return adj
