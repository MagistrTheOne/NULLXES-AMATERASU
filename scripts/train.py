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
from amaterasu.data.hifi_batches import iter_circuit0_batches
from amaterasu.distributed.topologies import load_topology
from amaterasu.model.accounting import account, assert_frozen_total
from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.model.init import materialize_empty
from amaterasu.training.curriculum import TrainConfig
from amaterasu.training.loop import train_loop
from amaterasu.training.trainable import apply_trainable
from amaterasu.utils.logging import log


def main() -> None:
    p = argparse.ArgumentParser(description="AMATERASU-32B train loop")
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--mixture", default="research")
    p.add_argument("--topology", type=Path, default=ROOT / "configs" / "distributed" / "h200_141gb.json")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--hifi-dir", type=Path, default=None)
    p.add_argument("--circuit0", action="store_true", help="Stage-1 production data path: HiFi parquet, trainable=nces")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = Amaterasu32BConfig()
    topo = load_topology(args.topology)
    log(f"topology {topo.name} tp={topo.tp} pp={topo.pp} ep={topo.ep}")

    if args.max_steps > 0 and not args.circuit0:
        raise SystemExit("refusing empty-iterator train; pass --circuit0 --ckpt --hifi-dir")
    if args.circuit0:
        if args.ckpt is None or args.hifi_dir is None:
            raise SystemExit("--circuit0 requires --ckpt and --hifi-dir")
        if args.max_steps <= 0:
            args.max_steps = 1
        if args.stage != 1:
            raise SystemExit("circuit-0 is Stage 1 only")

    with torch.device("meta"):
        model = Amaterasu32B(cfg)
    report = account(model)
    assert_frozen_total(report)
    log(f"account total={report.total}")

    train_cfg = TrainConfig(
        stage=args.stage,
        mixture=args.mixture,
        max_steps=args.max_steps,
        circuit0=args.circuit0,
        log_every=1 if args.circuit0 else 20,
    )
    if args.max_steps <= 0:
        log("model identity verified on meta; pass --circuit0 --ckpt --hifi-dir --max-steps 1")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"materialize empty on {device}")
    materialize_empty(model, device)
    log(f"resume {args.ckpt}")
    resume_modules(model, args.ckpt, device=device, dtype=torch.bfloat16)
    n_train = apply_trainable(model)
    log(f"circuit0 trainable={n_train:,}")
    batches = iter_circuit0_batches(
        args.hifi_dir,
        batch_size=args.batch_size,
        max_episodes=64,
        max_rows=max(args.max_steps * args.batch_size, 8),
        device=device,
    )
    train_loop(model, batches, train_cfg)
    log("CIRCUIT0 TRAIN GATE PASSED")


if __name__ == "__main__":
    main()
