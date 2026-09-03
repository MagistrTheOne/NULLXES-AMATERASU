from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator

import torch

from amaterasu.data.hifi_schema import GRIP, LEFT, RIGHT, ROT6D, XYZ


def _as_20(x) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
    if t.numel() != 20:
        raise ValueError(f"HiFi state/action must be 20-d, got {tuple(t.shape)}")
    return t


def state20_to_hands(state) -> dict[str, torch.Tensor]:
    s = _as_20(state)
    right, left = s[RIGHT], s[LEFT]
    return {
        "right_xyz": right[XYZ].contiguous(),
        "right_rot6d": right[ROT6D].contiguous(),
        "right_grip": right[GRIP].reshape(()).contiguous(),
        "left_xyz": left[XYZ].contiguous(),
        "left_rot6d": left[ROT6D].contiguous(),
        "left_grip": left[GRIP].reshape(()).contiguous(),
    }


def iter_parquet_records(path: Path, max_episodes: int, max_rows: int) -> Iterator[dict]:
    import pyarrow.parquet as pq

    from amaterasu.data.hifi_schema import PARQUET_COLUMNS

    table = pq.read_table(path, columns=list(PARQUET_COLUMNS))
    seen: set[int] = set()
    rows = 0
    for i in range(table.num_rows):
        if rows >= max_rows:
            return
        rec = {c: table.column(c)[i].as_py() for c in PARQUET_COLUMNS}
        if rec.get("valid.frame") is False:
            continue
        ep = int(rec["episode_index"])
        if ep not in seen and len(seen) >= max_episodes:
            continue
        seen.add(ep)
        hands = state20_to_hands(rec["observation.state"])
        rows += 1
        yield {
            **hands,
            "episode_index": ep,
            "frame_index": int(rec["frame_index"]),
            "timestamp": float(rec["timestamp"] or 0.0),
            "episode_reset": int(rec["frame_index"]) == 0,
        }


def iter_hifi_dir(root: Path, max_episodes: int = 64, max_rows: int = 4096) -> Iterator[dict]:
    files = sorted(root.rglob("*.parquet")) if root.exists() else []
    if not files:
        raise FileNotFoundError(
            f"no parquet under {root}. "
            "Run: python3 scripts/materialize_hifi.py --output /workspace/data/hifi_cap "
            "--max-bytes 536870912 --max-files 2"
        )
    remaining_ep = max_episodes
    remaining_rows = max_rows
    seen_eps: set[tuple[str, int]] = set()
    for f in files:
        if remaining_ep <= 0 or remaining_rows <= 0:
            return
        for rec in iter_parquet_records(f, max_episodes=remaining_ep, max_rows=remaining_rows):
            key = (str(f), rec["episode_index"])
            if key not in seen_eps and len(seen_eps) >= max_episodes:
                continue
            seen_eps.add(key)
            remaining_rows -= 1
            yield rec
            if remaining_rows <= 0:
                return
        remaining_ep = max_episodes - len({k[1] for k in seen_eps})
