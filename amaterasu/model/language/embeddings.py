from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.language.special_tokens import NULL_INSTRUCTION_ID


class LanguageEmbeddings(nn.Module):
    """Untied wte + lm_head, plus 16 modality and 16 special rows. Vocab stays 65536."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.modality_emb = nn.Embedding(cfg.n_modality_emb, cfg.d_model)
        self.special_emb = nn.Embedding(cfg.n_special_emb, cfg.d_model)
        self.null_instruction_id = NULL_INSTRUCTION_ID

    def encode(
        self,
        input_ids: torch.Tensor,
        modality_ids: torch.Tensor,
        special_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.wte(input_ids)
        x = x + self.modality_emb(modality_ids.long().clamp(0, self.modality_emb.num_embeddings - 1))
        if special_ids is not None:
            valid = special_ids >= 0
            sp = self.special_emb(special_ids.clamp(min=0).long())
            x = x + sp * valid.unsqueeze(-1).to(x.dtype)
        return x

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden.float())

    def null_instruction(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch, 1), self.null_instruction_id, dtype=torch.long, device=device)
