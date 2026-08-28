from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Profiler:
    times: list[float] = field(default_factory=list)

    def record(self, dt: float) -> None:
        self.times.append(float(dt))

    def mean(self) -> float:
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)

    def tokens_per_sec(self, tokens: int) -> float:
        m = self.mean()
        if m <= 0:
            return 0.0
        return tokens / m
