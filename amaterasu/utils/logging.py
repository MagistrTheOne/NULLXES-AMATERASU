from __future__ import annotations

import sys
from datetime import datetime, timezone


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()
