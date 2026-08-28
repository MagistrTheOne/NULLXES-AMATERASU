from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.constants import FROZEN_TOTAL, LEDGER
from amaterasu.model.accounting import account, assert_frozen_total
from amaterasu.model.amaterasu import Amaterasu32B


def main() -> None:
    parser = argparse.ArgumentParser(description="AMATERASU-32B Phase II init and parameter accounting")
    parser.add_argument("--write-weights", action="store_true", help="stream-init sharded safetensors")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "amaterasu_32b_v0.1_init")
    args = parser.parse_args()

    cfg = Amaterasu32BConfig()
    cfg.to_json(ROOT / "configs" / "model" / "amaterasu_32b_v0.1.json")

    with torch.device("meta"):
        model = Amaterasu32B(cfg)

    report = account(model)
    print("TOTAL UNIVERSAL PARAMETERS")
    print(f"{report.total:,}")
    print(f"freeze_hash {cfg.freeze_hash()}")
    print("components:")
    for k in LEDGER:
        mark = "OK" if report.diffs[k] == 0 else f"DIFF {report.diffs[k]:+d}"
        print(f"  {k:24s} {report.components[k]:>15,}  {mark}")
    print("graphs:")
    for k, v in report.graphs.items():
        print(f"  {k:28s} {v:>15,}")
    assert_frozen_total(report)
    if report.total != FROZEN_TOTAL:
        raise SystemExit("PHASE II FAILS")
    print("PHASE II ACCOUNTING GATE PASSED")

    if args.write_weights:
        from amaterasu.checkpoint.safetensors_io import stream_init_safetensors

        stream_init_safetensors(model, args.out)
        print(f"wrote safetensors shards to {args.out}")


if __name__ == "__main__":
    main()
