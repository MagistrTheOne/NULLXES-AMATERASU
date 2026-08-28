from __future__ import annotations

import torch

from amaterasu.tensors.ecd_schema import D_ECD


def pack_ecd(
    topology_class: torch.Tensor,
    effectors: torch.Tensor,
    workspace: torch.Tensor,
    locomotion: torch.Tensor,
    manipulation: torch.Tensor,
    sensors: torch.Tensor,
    payload: torch.Tensor,
    dexterity: torch.Tensor,
    mobility: torch.Tensor,
) -> torch.Tensor:
    """Pack capability fields into [B, 128]. No motor constants or gains."""
    parts = [
        _fit(topology_class, 16),
        _fit(effectors, 16),
        _fit(workspace, 16),
        _fit(locomotion, 8),
        _fit(manipulation, 8),
        _fit(sensors, 16),
        _fit(payload, 8),
        _fit(dexterity, 8),
        _fit(mobility, 16),
        topology_class.new_zeros(topology_class.shape[0], 16),
    ]
    return torch.cat(parts, dim=-1)


def _fit(x: torch.Tensor, width: int) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    b = x.shape[0]
    flat = x.reshape(b, -1).float()
    out = flat.new_zeros(b, width)
    n = min(flat.shape[-1], width)
    out[:, :n] = flat[:, :n]
    return out


def empty_topo(batch: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch, 32, 32, device=device)


def humanoid_ecd(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    topo = torch.zeros(batch, 16, device=device)
    topo[:, 0] = 1.0
    effectors = torch.zeros(batch, 16, device=device)
    effectors[:, :4] = 1.0
    workspace = torch.zeros(batch, 16, device=device)
    workspace[:, 0] = 1.0
    loco = torch.zeros(batch, 8, device=device)
    loco[:, 0] = 1.0
    manip = torch.zeros(batch, 8, device=device)
    manip[:, 0] = 1.0
    sensors = torch.zeros(batch, 16, device=device)
    sensors[:, 0] = 1.0
    payload = torch.zeros(batch, 8, device=device)
    dex = torch.zeros(batch, 8, device=device)
    dex[:, 1] = 1.0
    mobility = torch.zeros(batch, 16, device=device)
    raw = pack_ecd(topo, effectors, workspace, loco, manip, sensors, payload, dex, mobility)
    if raw.shape[-1] != D_ECD:
        raise RuntimeError("ECD pack width drifted from D_ECD")
    grid = empty_topo(batch, device)
    for i in range(min(9, 32)):
        grid[:, i, i] = 1.0
        if i + 1 < 9:
            grid[:, i, i + 1] = 1.0
            grid[:, i + 1, i] = 1.0
    return raw, grid
