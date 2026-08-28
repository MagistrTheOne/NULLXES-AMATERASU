from __future__ import annotations

from dataclasses import dataclass

import torch

from amaterasu.tensors.modality import ModalityId
from amaterasu.tensors.z_schema import N_DYN, N_HUM, N_OBJ, N_SCENE


@dataclass
class ZPack:
    """Pooling views over HPT/NCES tokens. No extra embedding tables."""

    z_self: torch.Tensor
    z_contact: torch.Tensor
    z_objects: torch.Tensor
    z_humans: torch.Tensor
    z_scene: torch.Tensor
    z_dynamics: torch.Tensor
    z_uncertainty: torch.Tensor
    packed: torch.Tensor
    kind: torch.Tensor
    mask: torch.Tensor


def pool_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(tokens.dtype).unsqueeze(-1)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (tokens * w).sum(dim=1) / denom


def contact_from_edges(
    nces_tokens: torch.Tensor,
    contact_idx: torch.Tensor,
    contact_valid: torch.Tensor,
) -> torch.Tensor:
    """Sparse contacts [B,E,2] → [B,E,d] by summing the two node tokens. Parameter-free."""
    b, e, _ = contact_idx.shape
    d = nces_tokens.shape[-1]
    out = nces_tokens.new_zeros(b, e, d)
    idx = contact_idx.clamp(min=0)
    n = nces_tokens.shape[1]
    idx = idx.clamp(max=max(n - 1, 0))
    i = idx[..., 0]
    j = idx[..., 1]
    bi = torch.arange(b, device=nces_tokens.device).unsqueeze(-1).expand(b, e)
    a = nces_tokens[bi, i]
    c = nces_tokens[bi, j]
    out = (a + c) * 0.5
    return out * contact_valid.unsqueeze(-1).to(out.dtype)


def pack_z(
    hpt: torch.Tensor,
    modality_ids: torch.Tensor,
    nces_tokens: torch.Tensor,
    nces_valid: torch.Tensor,
    aux9: torch.Tensor,
    contact_idx: torch.Tensor | None = None,
    contact_valid: torch.Tensor | None = None,
    n_obj: int = N_OBJ,
    n_hum: int = N_HUM,
    n_scene: int = N_SCENE,
    n_dyn: int = N_DYN,
) -> ZPack:
    vis = modality_ids == int(ModalityId.VISION)
    lang = modality_ids == int(ModalityId.LANGUAGE)
    phys = modality_ids == int(ModalityId.PHYSICAL)
    z_self = nces_tokens * nces_valid.unsqueeze(-1).to(nces_tokens.dtype)
    z_scene = _split_slots(hpt, vis, n_scene)
    z_objects = _split_slots(hpt, vis, n_obj)
    z_humans = _split_slots(hpt, lang, n_hum)
    z_dyn = _split_slots(hpt, phys, n_dyn)
    if contact_idx is None:
        z_contact = nces_tokens.new_zeros(nces_tokens.shape[0], 1, nces_tokens.shape[-1])
        cvalid = torch.zeros(nces_tokens.shape[0], 1, dtype=torch.bool, device=hpt.device)
    else:
        assert contact_valid is not None
        z_contact = contact_from_edges(nces_tokens, contact_idx, contact_valid)
        cvalid = contact_valid
    z_unc = aux9
    packed, kind, mask = _concat_views(z_self, nces_valid, z_contact, cvalid, z_objects, z_humans, z_scene, z_dyn)
    return ZPack(z_self, z_contact, z_objects, z_humans, z_scene, z_dyn, z_unc, packed, kind, mask)


def _concat_views(
    z_self: torch.Tensor,
    self_valid: torch.Tensor,
    z_contact: torch.Tensor,
    contact_valid: torch.Tensor,
    z_objects: torch.Tensor,
    z_humans: torch.Tensor,
    z_scene: torch.Tensor,
    z_dyn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = z_self.shape[0]
    device = z_self.device
    parts = [z_self, z_contact, z_objects, z_humans, z_scene, z_dyn]
    packed = torch.cat(parts, dim=1)
    kinds = []
    masks = [self_valid, contact_valid]
    for k, t in enumerate((z_objects, z_humans, z_scene, z_dyn), start=2):
        kinds.append(torch.full((b, t.shape[1]), k, dtype=torch.int8, device=device))
        masks.append(torch.ones(b, t.shape[1], dtype=torch.bool, device=device))
    kind = torch.cat(
        [
            torch.zeros(b, z_self.shape[1], dtype=torch.int8, device=device),
            torch.ones(b, z_contact.shape[1], dtype=torch.int8, device=device),
            *kinds,
        ],
        dim=1,
    )
    mask = torch.cat(masks, dim=1)
    return packed, kind, mask


def _split_slots(tokens: torch.Tensor, mask: torch.Tensor, n_slots: int) -> torch.Tensor:
    b, s, d = tokens.shape
    if s == 0:
        return tokens.new_zeros(b, n_slots, d)
    idx = torch.arange(s, device=tokens.device) % n_slots
    out = tokens.new_zeros(b, n_slots, d)
    counts = tokens.new_zeros(b, n_slots, 1)
    wmask = mask.to(tokens.dtype)
    for slot in range(n_slots):
        m = wmask * (idx == slot).to(tokens.dtype)
        w = m.unsqueeze(-1)
        out[:, slot] = (tokens * w).sum(dim=1)
        counts[:, slot] = w.sum(dim=1)
    return out / counts.clamp_min(1.0)
