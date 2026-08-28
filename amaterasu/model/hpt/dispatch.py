from __future__ import annotations

import torch
from torch import nn

from amaterasu.tensors.modality import ModalityId


def apply_modality_ffn(
    x: torch.Tensor,
    modality_ids: torch.Tensor,
    norms: dict[int, nn.Module],
    ffns: dict[int, nn.Module],
) -> torch.Tensor:
    """FFN residual delta. Unselected modalities contribute zeros."""
    b, s, d = x.shape
    delta = x.new_zeros(b, s, d)
    flat_x = x.reshape(b * s, d)
    flat_d = delta.reshape(b * s, d)
    ids = modality_ids.reshape(b * s)
    for mod, ffn in ffns.items():
        idx = (ids == int(mod)).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        tok = ffn(norms[mod](flat_x.index_select(0, idx)))
        flat_d.index_copy_(0, idx, tok)
    return delta


def physical_token_mask(modality_ids: torch.Tensor) -> torch.Tensor:
    return modality_ids == int(ModalityId.PHYSICAL)
