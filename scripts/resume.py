from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from amaterasu.checkpoint.resume import resume_modules
from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.accounting import account, assert_frozen_total
from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.utils.logging import log


def main() -> None:
    p = argparse.ArgumentParser(description="Resume AMATERASU-32B from safetensors shards")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", default="meta")
    args = p.parse_args()
    cfg = Amaterasu32BConfig()
    if args.device == "meta":
        with torch.device("meta"):
            model = Amaterasu32B(cfg)
        report = account(model)
        assert_frozen_total(report)
        log(f"meta resume check total={report.total} ckpt={args.ckpt}")
        return
    model = Amaterasu32B(cfg)
    resume_modules(model, args.ckpt)
    log(f"loaded shards from {args.ckpt}")


if __name__ == "__main__":
    main()
