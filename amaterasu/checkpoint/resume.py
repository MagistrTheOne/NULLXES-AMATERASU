from __future__ import annotations

import json
from pathlib import Path

import torch

from amaterasu.checkpoint.safetensors_io import load_into_module
from amaterasu.model.amaterasu import Amaterasu32B


def resume_modules(
    model: Amaterasu32B,
    ckpt_dir: Path,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("frozen_total") != 31_740_290_560:
        raise RuntimeError("checkpoint frozen_total does not match AMATERASU-32B v0.1")
    mapping = {
        "embeddings": model.embeddings,
        "vision": model.vision,
        "audio": model.audio,
        "nces": model.nces,
        "tactile": model.tactile,
        "ecd": model.ecd,
        "ssm": model.ssm,
        "memory": model.memory,
        "dynamics": model.dynamics,
        "eac": model.eac,
        "flow": model.flow,
    }
    for name, module in mapping.items():
        path = ckpt_dir / f"{name}.safetensors"
        if path.exists():
            load_into_module(module, path, strict=False, device=device, dtype=dtype)
    for i, layer in enumerate(model.hpt.fast_layers):
        path = ckpt_dir / f"hpt-fast-{i}.safetensors"
        if path.exists():
            load_into_module(layer, path, strict=False, device=device, dtype=dtype)
    for i, layer in enumerate(model.hpt.slow_dense_layers):
        path = ckpt_dir / f"hpt-slow_dense-{i}.safetensors"
        if path.exists():
            load_into_module(layer, path, strict=False, device=device, dtype=dtype)
    for i, layer in enumerate(model.hpt.slow_moe_layers):
        path = ckpt_dir / f"hpt-slow_moe-{i}.safetensors"
        if path.exists():
            load_into_module(layer, path, strict=False, device=device, dtype=dtype)
