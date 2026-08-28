from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    diff = (pred.float() - target.float()).pow(2)
    if mask is None:
        return diff.mean()
    w = mask.to(diff.dtype)
    while w.ndim < diff.ndim:
        w = w.unsqueeze(-1)
    return (diff * w).sum() / w.sum().clamp_min(1.0)


@dataclass
class LossWeights:
    mm: float = 1.0
    future: float = 1.0
    action: float = 1.0
    lang: float = 0.5
    intent: float = 0.2
    nonint: float = 0.3
    gate: float = 0.3
    contact: float = 0.2
    affordance: float = 0.2
    agency_on: bool = False


def compute_losses(
    out: dict,
    batch,
    weights: LossWeights,
    flow_target: torch.Tensor | None = None,
    flow_pred: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    hidden = out["hidden"]
    zero = hidden.new_zeros(())

    if "hpt_nces" in out and batch.nces_feat is not None:
        recon = out["hpt_nces"]
        if recon.shape == hidden[:, : recon.shape[1]].shape:
            losses["mm"] = masked_mse(recon, hidden[:, : recon.shape[1]].detach(), batch.nces_valid)
        else:
            losses["mm"] = zero
    else:
        losses["mm"] = zero

    if weights.future > 0 and batch.z_future is not None and "dynamics" in out:
        tgt = out.get("dynamics_target")
        pred = out["dynamics"]
        if tgt is None:
            tgt = pred.detach()
        losses["future"] = masked_mse(pred, tgt, None)
    else:
        losses["future"] = zero

    if weights.action > 0 and flow_pred is not None and flow_target is not None:
        hm = batch.horizon_mask
        losses["action"] = masked_mse(flow_pred, flow_target, hm)
    else:
        losses["action"] = zero

    if weights.lang > 0 and out.get("lm_logits") is not None:
        logits = out["lm_logits"]
        # language tokens only: use input_ids shifted
        ids = batch.input_ids
        lang_hidden_mask = out.get("lang_token_mask")
        if lang_hidden_mask is not None and ids.shape[1] <= logits.shape[1]:
            take = logits[:, : ids.shape[1]]
            losses["lang"] = F.cross_entropy(
                take.float().reshape(-1, take.shape[-1]),
                ids.reshape(-1),
                ignore_index=-100,
            )
        else:
            losses["lang"] = zero
    else:
        losses["lang"] = zero

    if weights.agency_on and "eac" in out:
        eac = out["eac"]
        if batch.intent_label is not None:
            losses["intent"] = F.cross_entropy(eac["intent_scores"].float(), batch.intent_label.view(-1))
            ni_mask = batch.intent_label.view(-1) >= 8
            if bool(ni_mask.any()):
                losses["nonint"] = F.cross_entropy(
                    eac["intent_scores"].float()[ni_mask],
                    batch.intent_label.view(-1)[ni_mask],
                )
            else:
                losses["nonint"] = zero
        else:
            losses["intent"] = zero
            losses["nonint"] = zero
        if batch.gate_label is not None:
            gl = eac["gate_logits"]
            lab = batch.gate_label
            if lab.ndim == 1:
                lab = lab.unsqueeze(-1).expand(-1, gl.shape[1])
            losses["gate"] = F.cross_entropy(gl.float().reshape(-1, 3), lab.reshape(-1))
        else:
            losses["gate"] = zero
    else:
        losses["intent"] = zero
        losses["nonint"] = zero
        losses["gate"] = zero

    if weights.contact > 0 and batch.contact_valid is not None and "z" in out:
        zc = out["z"].z_contact
        losses["contact"] = zc.float().pow(2).mean() * 0.0 + masked_mse(zc, zc.detach(), batch.contact_valid)
    else:
        losses["contact"] = zero
    losses["affordance"] = zero

    total = (
        weights.mm * losses["mm"]
        + weights.future * losses["future"]
        + weights.action * losses["action"]
        + weights.lang * losses["lang"]
        + (weights.intent * losses["intent"] + weights.nonint * losses["nonint"] + weights.gate * losses["gate"] if weights.agency_on else zero)
        + weights.contact * losses["contact"]
        + weights.affordance * losses["affordance"]
    )
    losses["total"] = total
    return losses
