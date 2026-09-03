from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amaterasu.data.hifi_batches import iter_circuit0_batches


def main() -> None:
    p = argparse.ArgumentParser(description="CPU decode gate: parquet → AMATERASUBatch, no GPU")
    p.add_argument("--hifi-dir", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=8)
    args = p.parse_args()
    n = 0
    for batch in iter_circuit0_batches(args.hifi_dir, batch_size=1, max_episodes=64, max_rows=args.max_rows):
        n += 1
        print(
            f"ok batch={n} source={batch.source_ids} nces={tuple(batch.nces_feat.shape)} "
            f"intent={batch.intent_label} video={batch.video is not None}"
        )
    if n == 0:
        raise SystemExit("AMATERASU DATA GATE FAILED: zero batches")
    print(f"HIFI DECODE GATE PASSED batches={n}")


if __name__ == "__main__":
    main()
