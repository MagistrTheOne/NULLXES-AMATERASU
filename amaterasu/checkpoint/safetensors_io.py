from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from amaterasu.constants import FROZEN_TOTAL, MODEL_ID
from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.model.init import init_parameters, materialize_cpu


def shard_modules(model: Amaterasu32B) -> list[tuple[str, nn.Module, int]]:
    shards: list[tuple[str, nn.Module, int]] = [
        ("embeddings", model.embeddings, 40),
        ("vision", model.vision, 36),
        ("audio", model.audio, 8),
        ("nces", model.nces, 6),
        ("tactile", model.tactile, 1),
        ("ecd", model.ecd, 3),
        ("ssm", model.ssm, 4),
        ("memory", model.memory, 2),
        ("dynamics", model.dynamics, 8),
        ("eac", model.eac, 4),
        ("flow", model.flow, 12),
    ]
    for i, layer in enumerate(model.hpt.fast_layers):
        shards.append((f"hpt.fast.{i}", layer, 40))
    for i, layer in enumerate(model.hpt.slow_dense_layers):
        shards.append((f"hpt.slow_dense.{i}", layer, 40))
    for i, layer in enumerate(model.hpt.slow_moe_layers):
        shards.append((f"hpt.slow_moe.{i}", layer, 40))
    return shards


def stream_init_safetensors(model: Amaterasu32B, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for name, module, depth in shard_modules(model):
        materialize_cpu(module)
        init_parameters(module, depth)
        tensors = {k: v.contiguous() for k, v in module.state_dict().items()}
        path = out_dir / f"{name.replace('.', '-')}.safetensors"
        print(f"init+save {name} ({sum(t.numel() for t in tensors.values()):,} tensors)", flush=True)
        save_file(tensors, str(path))
        for k in tensors:
            index[f"{name}.{k}"] = path.name
        module.to_empty(device=torch.device("meta"))
        del tensors
    manifest = {
        "format": "amaterasu-ckpt-v1",
        "model_id": MODEL_ID,
        "frozen_total": FROZEN_TOTAL,
        "freeze_hash": model.cfg.freeze_hash(),
        "weight_map": index,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_shard(
    path: Path,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    tensors = load_file(str(path))
    out: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if dtype is not None and v.is_floating_point():
            v = v.to(dtype=dtype)
        if device is not None:
            v = v.to(device=device)
        out[k] = v
    return out


def load_into_module(
    module: nn.Module,
    path: Path,
    strict: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    sd = load_shard(path, device=device, dtype=dtype)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"load mismatch missing={missing} unexpected={unexpected}")


def save_optimizer(state: dict, path: Path) -> None:
    tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
    meta = {k: v for k, v in state.items() if not torch.is_tensor(v)}
    save_file(tensors, str(path))
    path.with_suffix(".json").write_text(json.dumps(meta, default=str), encoding="utf-8")
