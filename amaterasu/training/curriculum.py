from __future__ import annotations

from dataclasses import dataclass

from amaterasu.training.losses import LossWeights

STAGES = {
    1: "representation",
    2: "motion",
    3: "manipulation",
    4: "cross_embodiment",
    5: "robot",
    6: "dynamics",
    7: "agency",
    8: "autonomous",
    9: "sim2real",
}


def weights_for_stage(stage: int) -> LossWeights:
    if stage < 1 or stage > 9:
        raise ValueError("stage must be 1–9")
    w = LossWeights()
    if stage < 6:
        w.future = 0.0
    if stage < 2:
        w.action = 0.0
    if stage < 7:
        w.agency_on = False
        w.intent = 0.0
        w.nonint = 0.0
        w.gate = 0.0
    else:
        w.agency_on = True
    if stage == 1:
        w.lang = 0.5
        w.action = 0.0
        w.future = 0.0
    if stage >= 6:
        w.future = 1.0
    if stage >= 2:
        w.action = 1.0
    return w


@dataclass
class TrainConfig:
    stage: int = 1
    global_batch_tokens: int = 4_000_000
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100_000
    nfe_fast: int = 1
    nfe_precision: int = 4
    s_max: int = 8192
    mixture: str = "research"
    seed: int = 0
    log_every: int = 20
    ckpt_every: int = 500
    circuit0: bool = False
