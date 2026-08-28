from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from amaterasu.constants import (
    EPISODIC_RING_SIZE,
    FROZEN_TOTAL,
    H_CHUNK_MAX,
    MODEL_ID,
    N_CAM_MAX,
    N_NODES_MAX,
    S_MAX,
    SEMANTIC_RING_SIZE,
    T_CLIP,
    TBPTT_FAST_TICKS,
    TRAIN_H,
    TRAIN_W,
    VOCAB_SIZE,
)


@dataclass(frozen=True)
class Amaterasu32BConfig:
    """Immutable architecture. Training flags do not live here."""

    name: str = MODEL_ID
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_head: int = 128
    d_ff: int = 11008
    d_ff_expert: int = 4096
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_topk: int = 2
    n_fast_layers: int = 12
    n_slow_layers: int = 28
    n_moe_layers: int = 20
    n_slow_dense_physical: int = 8
    vocab_size: int = VOCAB_SIZE
    untied_lm_head: bool = True
    qk_norm_shared: bool = True
    bias: bool = False
    n_modality_emb: int = 16
    n_special_emb: int = 16
    rms_eps: float = 1e-6
    rope_theta: float = 10_000.0
    ve_d: int = 2048
    ve_layers: int = 36
    ve_heads: int = 16
    ve_kv: int = 4
    ve_d_head: int = 128
    ve_d_ff: int = 5504
    tubelet: tuple[int, int, int, int] = (2, 14, 14, 3)
    n_cam_max: int = N_CAM_MAX
    t_clip: int = T_CLIP
    train_h: int = TRAIN_H
    train_w: int = TRAIN_W
    nces_layers: int = 6
    nces_d: int = 2048
    nces_in: int = 128
    n_nodes_max: int = N_NODES_MAX
    audio_d: int = 768
    audio_layers: int = 8
    audio_heads: int = 12
    audio_d_head: int = 64
    audio_d_ff: int = 2048
    audio_mel: int = 128
    tactile_in: int = 256
    ssm_layers: int = 4
    ssm_d_inner: int = 8192
    ssm_d_state: int = 128
    ssm_dt_rank: int = 256
    ssm_d_conv: int = 4
    wm_slots: int = 256
    dyn_layers: int = 8
    dyn_d: int = 3072
    dyn_heads: int = 24
    dyn_kv: int = 6
    dyn_d_head: int = 128
    dyn_d_ff: int = 8192
    eac_gcis_layers: int = 4
    eac_queries: int = 11
    eac_agency_tokens: int = 64
    eac_aux_heads: int = 9
    flow_layers: int = 12
    flow_d: int = 2048
    flow_heads: int = 16
    flow_kv: int = 4
    flow_d_head: int = 128
    flow_d_ff: int = 5504
    flow_io: int = 512
    flow_time_d: int = 256
    h_chunk_max: int = H_CHUNK_MAX
    ecd_in: int = 128
    ecd_hidden: int = 1024
    ecd_topo: int = 32
    ecd_d: int = 512
    ecd_d_ff: int = 1408
    ecd_layers: int = 3
    s_max: int = S_MAX
    attn_window: int = 4096
    episodic_ring_size: int = EPISODIC_RING_SIZE
    semantic_ring_size: int = SEMANTIC_RING_SIZE
    tbptt_fast_ticks: int = TBPTT_FAST_TICKS
    frozen_total: int = FROZEN_TOTAL
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("n_heads * d_head must equal d_model")
        if self.n_fast_layers + self.n_slow_layers != 40:
            raise ValueError("Fast+Slow must be 40 HPT layers")
        if self.n_slow_dense_physical + self.n_moe_layers != self.n_slow_layers:
            raise ValueError("Slow dense physical + MoE must equal Slow layers")
        if self.bias:
            raise ValueError("AMATERASU-32B v0.1 forbids HPT linear bias")

    def canonical_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tubelet"] = list(self.tubelet)
        d.pop("extra", None)
        return d

    def freeze_hash(self) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.canonical_dict()
        payload["freeze_hash"] = self.freeze_hash()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        path.with_suffix(".sha256").write_text(self.freeze_hash() + "\n", encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Amaterasu32BConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("freeze_hash", None)
        raw["tubelet"] = tuple(raw["tubelet"])
        extra = raw.pop("extra", {})
        cfg = Amaterasu32BConfig(**raw, extra=extra)
        return cfg
