from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClockConfig:
    slow_hz: float = 3.0
    fast_hz: float = 50.0
    nfe_fast: int = 1
    nfe_precision: int = 4
    tbptt_fast_ticks: int = 16

    @property
    def slow_period_s(self) -> float:
        return 1.0 / self.slow_hz

    @property
    def fast_period_s(self) -> float:
        return 1.0 / self.fast_hz


def ticks_until_slow(fast_tick: int, cfg: ClockConfig) -> bool:
    ratio = max(int(round(cfg.fast_hz / cfg.slow_hz)), 1)
    return fast_tick % ratio == 0
