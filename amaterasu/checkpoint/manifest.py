from __future__ import annotations

import json
from pathlib import Path

from amaterasu.constants import FROZEN_TOTAL, MODEL_ID


def write_manifest(path: Path, freeze_hash: str, extra: dict | None = None) -> None:
    payload = {
        "format": "amaterasu-ckpt-v1",
        "model_id": MODEL_ID,
        "frozen_total": FROZEN_TOTAL,
        "freeze_hash": freeze_hash,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
