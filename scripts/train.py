from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.distributed.topologies import load_topology
from amaterasu.model.accounting import account, assert_frozen_total
from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.training.curriculum import TrainConfig
from amaterasu.training.loop import train_loop
from amaterasu.utils.logging import log


def main() -> None:
    p = argparse.ArgumentParser(description="AMATERASU-32B train loop")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--mixture", default="research")
    p.add_argument("--topology", type=Path, default=ROOT / "configs" / "distributed" / "h200_141gb.json")
    p.add_argument("--max-steps", type=int, default=0)
    args = p.parse_args()
    cfg = Amaterasu32BConfig()
    topo = load_topology(args.topology)
    log(f"topology {topo.name} tp={topo.tp} pp={topo.pp} ep={topo.ep}")
    import torch

    with torch.device("meta"):
        model = Amaterasu32B(cfg)
    report = account(model)
    assert_frozen_total(report)
    log(f"account total={report.total}")
    train_cfg = TrainConfig(stage=args.stage, mixture=args.mixture, max_steps=args.max_steps)
    if args.max_steps <= 0:
        log("model identity verified on meta; pass --max-steps > 0 on a materialized rank to train")
        return
    train_loop(model, iter(()), train_cfg)


if __name__ == "__main__":
    main()
