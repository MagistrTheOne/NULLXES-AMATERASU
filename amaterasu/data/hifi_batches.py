from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from amaterasu.data.adapters.hifi_umi import record_to_sample
from amaterasu.data.collate import collate_samples
from amaterasu.data.hifi_reader import iter_hifi_dir
from amaterasu.data.validate_stage1 import validate_circuit0_batch, validate_circuit0_sample
from amaterasu.tensors.sample import AMATERASUBatch


def iter_circuit0_batches(
    root: Path,
    batch_size: int = 1,
    max_episodes: int = 64,
    max_rows: int = 4096,
    device: torch.device | None = None,
) -> Iterator[AMATERASUBatch]:
    buf = []
    for rec in iter_hifi_dir(root, max_episodes=max_episodes, max_rows=max_rows):
        cpu_rec = {
            "left_xyz": rec["left_xyz"],
            "left_rot6d": rec["left_rot6d"],
            "left_grip": rec["left_grip"],
            "right_xyz": rec["right_xyz"],
            "right_rot6d": rec["right_rot6d"],
            "right_grip": rec["right_grip"],
        }
        sample = record_to_sample(cpu_rec)
        sample.episode_reset = bool(rec["episode_reset"])
        validate_circuit0_sample(sample)
        if device is not None:
            sample.nces_feat = sample.nces_feat.to(device)
            sample.nces_valid = sample.nces_valid.to(device)
            sample.ecd_raw = sample.ecd_raw.to(device)
            sample.ecd_topo = sample.ecd_topo.to(device)
            sample.input_ids = sample.input_ids.to(device)
            sample.lang_mask = sample.lang_mask.to(device)
            if sample.node_mask is not None:
                sample.node_mask = sample.node_mask.to(device)
        buf.append(sample)
        if len(buf) >= batch_size:
            batch = collate_samples(buf)
            validate_circuit0_batch(batch)
            yield batch
            buf = []
    if buf:
        batch = collate_samples(buf)
        validate_circuit0_batch(batch)
        yield batch
