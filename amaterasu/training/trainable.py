from __future__ import annotations

from torch import nn

from amaterasu.model.amaterasu import Amaterasu32B

CIRCUIT0_TRAINABLE = ("nces",)
# NCES encoder is 274,490,880. Fail if someone expands the subset silently.
CIRCUIT0_MAX_TRAINABLE = 300_000_000


def apply_trainable(model: Amaterasu32B, prefixes: tuple[str, ...] = CIRCUIT0_TRAINABLE) -> int:
    for p in model.parameters():
        p.requires_grad = False
    for name in prefixes:
        mod = getattr(model, name)
        if not isinstance(mod, nn.Module):
            raise RuntimeError(f"trainable prefix {name} is not a module")
        for p in mod.parameters():
            p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n == 0:
        raise RuntimeError("trainable set is empty")
    if n > CIRCUIT0_MAX_TRAINABLE:
        raise RuntimeError(
            f"trainable {n:,} exceeds circuit-0 cap {CIRCUIT0_MAX_TRAINABLE:,}; "
            "full 32B AdamW is forbidden on this path"
        )
    return n
