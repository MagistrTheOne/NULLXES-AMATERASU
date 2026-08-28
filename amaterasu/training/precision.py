from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrecisionConfig:
    param_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype = torch.float32
    router_dtype: torch.dtype = torch.float32
    qtheta_dtype: torch.dtype = torch.float32


def cast_except_router(module: torch.nn.Module, dtype: torch.dtype) -> None:
    for name, p in module.named_parameters():
        if "router" in name or "gate.fc" in name or "q_theta" in name:
            p.data = p.data.float()
        else:
            p.data = p.data.to(dtype)
