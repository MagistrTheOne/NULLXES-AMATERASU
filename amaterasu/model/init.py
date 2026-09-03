from __future__ import annotations

import math

import torch
from torch import nn


def _is_norm_weight(name: str) -> bool:
    lname = name.lower()
    return (
        lname.endswith("weight")
        and (
            "norm" in lname
            or "q_norm" in lname
            or "k_norm" in lname
        )
    )


def _is_residual_out(name: str) -> bool:
    return name.endswith("wo.weight") or name.endswith("w_down.weight") or name.endswith("out_proj.weight")


def init_parameters(module: nn.Module, residual_layers: int) -> None:
    """From-scratch init. No pretrained loads. Meta tensors are skipped."""
    std_res = 0.02 / math.sqrt(2.0 * max(residual_layers, 1))
    for name, p in module.named_parameters(recurse=True):
        if p.device.type == "meta":
            continue
        if "A_log" in name:
            with torch.no_grad():
                p.copy_(torch.log(torch.arange(1, p.numel() + 1, device=p.device, dtype=p.dtype)))
            continue
        if p.ndim == 1:
            if _is_norm_weight(name) or name.endswith(".D") or name.endswith("D"):
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)
            continue
        std = std_res if _is_residual_out(name) else 0.02
        nn.init.normal_(p, mean=0.0, std=std)


def materialize_cpu(module: nn.Module) -> None:
    module.to_empty(device=torch.device("cpu"))


def materialize_empty(
    module: nn.Module,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> None:
    """Allocate storage on `device`. Cast dtype while still meta — never fp32-empty a 32B on GPU."""
    if dtype is not None:
        module.to(dtype=dtype)
    module.to_empty(device=device)
