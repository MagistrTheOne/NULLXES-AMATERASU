from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    name: str
    tp: int
    pp: int
    ep: int
    precision: str = "bf16"
    gpu_memory_gb: int | None = None

    def world_multiple(self) -> int:
        return self.tp * self.pp * self.ep
