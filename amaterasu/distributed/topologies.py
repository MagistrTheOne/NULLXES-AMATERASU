from __future__ import annotations

import json
from pathlib import Path

from amaterasu.distributed.mesh import Topology


def load_topology(path: Path) -> Topology:
    raw = json.loads(path.read_text(encoding="utf-8"))
    gb = raw.get("gpu_memory_gb")
    return Topology(
        name=str(raw["profile"]),
        tp=int(raw["tp"]),
        pp=int(raw["pp"]),
        ep=int(raw["ep"]),
        precision=str(raw.get("precision", "bf16")),
        gpu_memory_gb=int(gb) if gb is not None else None,
    )


def primary_h200() -> Topology:
    return load_topology(Path(__file__).resolve().parents[2] / "configs" / "distributed" / "h200_141gb.json")
