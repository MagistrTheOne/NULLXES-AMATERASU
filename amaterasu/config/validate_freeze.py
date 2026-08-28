from __future__ import annotations

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.constants import FROZEN_TOTAL, MODEL_ID, VOCAB_SIZE


def assert_frozen(cfg: Amaterasu32BConfig) -> None:
    if cfg.name != MODEL_ID:
        raise ValueError(f"config name {cfg.name!r} is not {MODEL_ID}")
    if cfg.frozen_total != FROZEN_TOTAL:
        raise ValueError("frozen_total mutated")
    if cfg.n_routed_experts != 8 or cfg.moe_topk != 2 or cfg.d_model != 4096:
        raise ValueError("architecture identity mutated; this is not AMATERASU-32B v0.1")
    if cfg.vocab_size != VOCAB_SIZE:
        raise ValueError("vocab_size mutated")
    if cfg.n_fast_layers != 12 or cfg.n_slow_layers != 28 or cfg.n_moe_layers != 20:
        raise ValueError("HPT layer split mutated")
