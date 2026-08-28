from __future__ import annotations

import torch
import torch.distributed as dist

from amaterasu.distributed.mesh import Topology


def expert_all_to_all(tokens: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """EP all-to-all. Identity when EP=1 or dist is uninitialized."""
    if group is None or not dist.is_available() or not dist.is_initialized():
        return tokens
    ws = dist.get_world_size(group)
    if ws == 1:
        return tokens
    chunks = list(tokens.chunk(ws, dim=0))
    out = [torch.empty_like(c) for c in chunks]
    dist.all_to_all(out, chunks, group=group)
    return torch.cat(out, dim=0)


def shard_experts(n_experts: int, topo: Topology, rank: int) -> slice:
    if topo.ep <= 1:
        return slice(0, n_experts)
    per = n_experts // topo.ep
    ep_rank = rank % topo.ep
    return slice(ep_rank * per, (ep_rank + 1) * per)


def assert_mesh(world_size: int, topo: Topology) -> None:
    m = topo.world_multiple()
    if world_size % m != 0:
        raise ValueError(f"world_size {world_size} is not divisible by TP*PP*EP={m}")
