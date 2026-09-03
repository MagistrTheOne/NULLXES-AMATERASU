from __future__ import annotations

import torch

from amaterasu.constants import N_NODES_MAX
from amaterasu.tensors.nces_schema import D_NCES_IN
from amaterasu.tensors.sample import AMATERASUSample, AMATERASUBatch


def validate_circuit0_sample(sample: AMATERASUSample) -> None:
    if sample.source != "hifi-umi-2k":
        raise RuntimeError(f"circuit-0 source must be hifi-umi-2k, got {sample.source!r}")
    if sample.intent_label is not None:
        raise RuntimeError("circuit-0 forbids intent_label (hand moved ≠ ACT)")
    if sample.gate_label is not None:
        raise RuntimeError("circuit-0 forbids gate_label")
    if sample.video is not None:
        raise RuntimeError("circuit-0 forbids video")
    if not sample.null_instruction:
        raise RuntimeError("circuit-0 requires null_instruction=True")
    if not sample.license_ok_commercial:
        raise RuntimeError("HiFi shard must be commercial-ok")
    if sample.nces_feat.shape[-1] != D_NCES_IN:
        raise RuntimeError(f"nces_feat last dim {sample.nces_feat.shape[-1]} != {D_NCES_IN}")
    if sample.nces_feat.shape[0] != N_NODES_MAX:
        raise RuntimeError(f"nces nodes {sample.nces_feat.shape[0]} != {N_NODES_MAX}")
    if not torch.isfinite(sample.nces_feat).all():
        raise RuntimeError("nces_feat is not finite")


def validate_circuit0_batch(batch: AMATERASUBatch) -> None:
    if batch.intent_label is not None:
        raise RuntimeError("circuit-0 batch has intent_label")
    if batch.video is not None:
        raise RuntimeError("circuit-0 batch has video")
    if not bool(batch.null_instruction.all()):
        raise RuntimeError("circuit-0 batch must be all null_instruction")
    if any(s != "hifi-umi-2k" for s in batch.source_ids):
        raise RuntimeError(f"circuit-0 mixed sources {batch.source_ids}")
