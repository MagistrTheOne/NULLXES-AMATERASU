from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.constants import EPISODIC_RING_SIZE, SEMANTIC_RING_SIZE
from amaterasu.model.memory.episodic import EpisodicMemory
from amaterasu.model.memory.semantic import SemanticMemory
from amaterasu.model.memory.working import WorkingMemory


@dataclass
class MemoryState:
    wm: torch.Tensor
    epi_ring: torch.Tensor
    epi_valid: torch.Tensor
    epi_cursor: torch.Tensor
    sem_keys: torch.Tensor
    sem_values: torch.Tensor
    sem_valid: torch.Tensor
    sem_counts: torch.Tensor
    sem_cursor: torch.Tensor

    def reset(self, episode_reset: torch.Tensor) -> "MemoryState":
        if not bool(episode_reset.any()):
            return self
        m = episode_reset.view(-1, 1, 1).to(self.wm.dtype)
        m2 = episode_reset.view(-1, 1)
        return MemoryState(
            wm=self.wm * (1.0 - m),
            epi_ring=self.epi_ring * (1.0 - m),
            epi_valid=self.epi_valid & ~episode_reset.view(-1, 1),
            epi_cursor=self.epi_cursor * (~episode_reset).long(),
            sem_keys=self.sem_keys * (1.0 - episode_reset.view(-1, 1, 1).to(self.sem_keys.dtype)),
            sem_values=self.sem_values * (1.0 - episode_reset.view(-1, 1, 1).to(self.sem_values.dtype)),
            sem_valid=self.sem_valid & ~episode_reset.view(-1, 1),
            sem_counts=self.sem_counts * (~episode_reset).view(-1, 1).to(self.sem_counts.dtype),
            sem_cursor=self.sem_cursor * (~episode_reset).long(),
        )


def empty_memory_state(batch: int, d_model: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
    return MemoryState(
        wm=torch.zeros(batch, 256, d_model, device=device, dtype=dtype),
        epi_ring=torch.zeros(batch, EPISODIC_RING_SIZE, d_model, device=device, dtype=dtype),
        epi_valid=torch.zeros(batch, EPISODIC_RING_SIZE, dtype=torch.bool, device=device),
        epi_cursor=torch.zeros(batch, dtype=torch.long, device=device),
        sem_keys=torch.zeros(batch, SEMANTIC_RING_SIZE, 512, device=device, dtype=dtype),
        sem_values=torch.zeros(batch, SEMANTIC_RING_SIZE, 512, device=device, dtype=dtype),
        sem_valid=torch.zeros(batch, SEMANTIC_RING_SIZE, dtype=torch.bool, device=device),
        sem_counts=torch.zeros(batch, SEMANTIC_RING_SIZE, device=device, dtype=dtype),
        sem_cursor=torch.zeros(batch, dtype=torch.long, device=device),
    )


def write_gate(
    aux9: torch.Tensor,
    contact: torch.Tensor,
    intent_switch: torch.Tensor,
    speech: torch.Tensor,
) -> torch.Tensor:
    """Gated write: surprise / contact / intent-switch / speech. Not every Fast tick."""
    novelty = aux9[:, 1] if aux9.shape[-1] > 1 else aux9[:, 0]
    uncertainty = aux9[:, 0]
    surprise = torch.sigmoid(novelty) + torch.sigmoid(uncertainty)
    return ((surprise > 1.0) | contact.bool() | intent_switch.bool() | speech.bool())


class MemorySystem(nn.Module):
    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.working = WorkingMemory(cfg)
        self.episodic = EpisodicMemory(cfg)
        self.semantic = SemanticMemory(cfg)
        self.d_model = cfg.d_model

    def new_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
        return empty_memory_state(batch, self.d_model, device, dtype)

    def write(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        state: MemoryState,
        aux9: torch.Tensor,
        contact: torch.Tensor,
        intent_switch: torch.Tensor,
        speech: torch.Tensor,
        query: torch.Tensor,
    ) -> MemoryState:
        wm = self.working(hidden, position_ids)
        do_write = write_gate(aux9, contact, intent_switch, speech)
        _, compressed = self.episodic.compress(hidden)
        epi_ring, epi_valid, epi_cursor = self.episodic.write_ring(
            state.epi_ring, state.epi_valid, state.epi_cursor, compressed, do_write
        )
        event = hidden.mean(dim=1)
        sem_k, sem_v, sem_valid, sem_counts, sem_cursor = self.semantic.write_ring(
            state.sem_keys,
            state.sem_values,
            state.sem_valid,
            state.sem_counts,
            state.sem_cursor,
            event,
            do_write,
        )
        return MemoryState(wm, epi_ring, epi_valid, epi_cursor, sem_k, sem_v, sem_valid, sem_counts, sem_cursor)

    def retrieve(self, state: MemoryState, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        epi = self.episodic.read(state.epi_ring, state.epi_valid, query)
        sem = self.semantic.retrieve(query, state.sem_keys, state.sem_values, state.sem_valid)
        return epi, sem

    def compress_evict(self, state: MemoryState) -> MemoryState:
        """Episodic overflow already evicts oldest; semantic evicts lowest retrieval count on write."""
        return state
