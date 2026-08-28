from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.inference.clocks import ClockConfig, ticks_until_slow
from amaterasu.inference.fast_loop import fast_tick, sensor_refresh
from amaterasu.inference.slow_loop import slow_tick
from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.tensors.sample import AMATERASUBatch
from amaterasu.utils.logging import log


def _toy_batch(device: torch.device) -> AMATERASUBatch:
    b, n = 1, 64
    return AMATERASUBatch(
        nces_feat=torch.zeros(b, n, 128, device=device),
        nces_valid=torch.zeros(b, n, dtype=torch.bool, device=device),
        ecd_raw=torch.zeros(b, 128, device=device),
        ecd_topo=torch.zeros(b, 32, 32, device=device),
        input_ids=torch.zeros(b, 1, dtype=torch.long, device=device),
        lang_mask=torch.ones(b, 1, dtype=torch.bool, device=device),
        node_mask=torch.zeros(b, n, dtype=torch.bool, device=device),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Dual-clock AMATERASU inference driver")
    p.add_argument("--fast-ticks", type=int, default=16)
    args = p.parse_args()
    clocks = ClockConfig()
    log(f"slow={clocks.slow_hz}Hz fast={clocks.fast_hz}Hz nfe_fast={clocks.nfe_fast}")
    cfg = Amaterasu32BConfig()
    with torch.device("meta"):
        model = Amaterasu32B(cfg)
    log("constructed AMATERASU-32B on meta for identity; dual-clock stepping uses fast_tick/slow_tick APIs")
    for t in range(args.fast_ticks):
        if ticks_until_slow(t, clocks):
            log(f"tick={t} SLOW_AGENCY")
        else:
            log(f"tick={t} FAST")
    log("modes: FAST_OBSERVE FAST_WAIT FAST_HOLD FAST_ACT FAST_SENSOR_REFRESH SLOW_AGENCY")


if __name__ == "__main__":
    main()
