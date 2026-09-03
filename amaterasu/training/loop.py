from __future__ import annotations

import itertools
import time
from collections.abc import Iterator

import torch

from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.model.accounting import account, assert_frozen_total
from amaterasu.model.flow.matching import pack_nces_action
from amaterasu.tensors.sample import AMATERASUBatch
from amaterasu.training.curriculum import TrainConfig, weights_for_stage
from amaterasu.training.losses import compute_losses
from amaterasu.utils.logging import log
from amaterasu.utils.profile import Profiler


def build_optimizer(model: Amaterasu32B, cfg: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    if not decay and not no_decay:
        raise RuntimeError("optimizer has zero trainable parameters")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
    )


def train_step(
    model: Amaterasu32B,
    batch: AMATERASUBatch,
    opt: torch.optim.AdamW,
    cfg: TrainConfig,
) -> dict[str, float]:
    model.train()
    weights = weights_for_stage(cfg.stage)
    if cfg.circuit0:
        weights.lang = 0.0
        weights.action = 0.0
        weights.future = 0.0
        weights.agency_on = False
    emit_lang = weights.lang > 0 and not cfg.circuit0
    use_flow = (not cfg.circuit0) and batch.nces_traj is not None and weights.action > 0
    xt = ut = pred_u = None
    flow_t = None
    if use_flow:
        assert batch.nces_traj is not None
        assert batch.node_mask is not None
        x1 = pack_nces_action(batch.nces_traj, batch.node_mask)
        x0 = torch.randn_like(x1)
        flow_t = torch.rand(x1.shape[0], device=x1.device)
        xt, ut = model.flow.interpolate(x0, x1, flow_t)
        hm = (
            batch.horizon_mask
            if batch.horizon_mask is not None
            else torch.ones(x1.shape[0], x1.shape[1], dtype=torch.bool, device=x1.device)
        )
        pred_u = model.flow.vector_field(xt, hm, flow_t)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=batch.nces_feat.is_cuda):
        out = model.forward_mode(
            "TRAIN",
            nces_feat=batch.nces_feat,
            nces_valid=batch.nces_valid,
            ecd_raw=batch.ecd_raw,
            ecd_topo=batch.ecd_topo,
            input_ids=batch.input_ids,
            lang_mask=batch.lang_mask,
            tactile=batch.tactile,
            tactile_valid=batch.tactile_valid,
            video=batch.video,
            camera_valid_mask=batch.camera_valid_mask,
            frame_times=batch.frame_times,
            audio_mel=batch.audio_mel,
            audio_mask=batch.audio_mask,
            nces_traj=batch.nces_traj if use_flow else None,
            horizon_mask=batch.horizon_mask,
            node_mask=batch.node_mask,
            flow_t=flow_t,
            emit_language=emit_lang,
            episode_reset=batch.episode_reset,
        )
        if "dynamics" in out and batch.z_future is not None:
            out["dynamics_target"] = model.dynamics.encode_target(batch.z_future)
        losses = compute_losses(out, batch, weights, flow_target=ut, flow_pred=pred_u)
    opt.zero_grad(set_to_none=True)
    losses["total"].backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    return {k: float(v.detach()) for k, v in losses.items()}


def train_loop(
    model: Amaterasu32B,
    batches: Iterator[AMATERASUBatch],
    cfg: TrainConfig,
) -> None:
    report = account(model)
    assert_frozen_total(report)
    try:
        first = next(batches)
    except StopIteration as exc:
        raise RuntimeError("AMATERASU DATA GATE FAILED: zero training batches") from exc
    batches = itertools.chain((first,), batches)
    opt = build_optimizer(model, cfg)
    prof = Profiler()
    step = 0
    seen = 0
    for batch in batches:
        seen += 1
        t0 = time.perf_counter()
        stats = train_step(model, batch, opt, cfg)
        dt = time.perf_counter() - t0
        prof.record(dt)
        step += 1
        if step % cfg.log_every == 0 or cfg.circuit0:
            log(f"step={step} loss={stats['total']:.6f} mm={stats['mm']:.6f} dt={dt:.3f}s")
        if not torch.isfinite(torch.tensor(stats["total"])):
            raise RuntimeError(f"non-finite loss at step={step}: {stats}")
        if step >= cfg.max_steps:
            break
    if seen == 0:
        raise RuntimeError("AMATERASU DATA GATE FAILED: zero training batches")
