from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class DataConfig:
    mixture: str = "research"
    commercial: bool = False
    num_workers: int = 0


def dump_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
