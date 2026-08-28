from __future__ import annotations

import torch

from amaterasu.tensors.modality import GateDecision, IntentKind


def select_intent(
    scores: torch.Tensor,
    gate_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Layer-2 gate then Qθ among ALLOW. If all ACT are BLOCK/DEFER, emit NOOP family.

    kinds 0–7 ACT, 8 OBSERVE, 9 HOLD, 10 WAIT.
    """
    decision = gate_logits.argmax(dim=-1)
    allow = decision == int(GateDecision.ALLOW)
    k = scores.shape[-1]
    kinds = torch.arange(k, device=scores.device).view(1, k)
    is_act = kinds < 8
    is_noop = kinds >= 8
    neg = torch.finfo(scores.dtype).min
    act_scores = scores.masked_fill(~(allow & is_act), neg)
    noop_scores = scores.masked_fill(~(allow & is_noop), neg)
    best_act = act_scores.argmax(dim=-1)
    best_noop = noop_scores.argmax(dim=-1)
    any_act_allow = (allow & is_act).any(dim=-1)
    chosen = torch.where(any_act_allow, best_act, best_noop)
    any_noop_allow = (allow & is_noop).any(dim=-1)
    fallback = torch.full_like(chosen, int(IntentKind.HOLD))
    chosen = torch.where(any_act_allow | any_noop_allow, chosen, fallback)
    chosen_kind = chosen.to(dtype=torch.int64)
    flow_on = (chosen_kind < 8) | (chosen_kind == int(IntentKind.HOLD))
    observe = chosen_kind == int(IntentKind.OBSERVE)
    hold = chosen_kind == int(IntentKind.HOLD)
    wait = chosen_kind == int(IntentKind.WAIT)
    act = chosen_kind < 8
    return {
        "chosen_kind": chosen_kind,
        "gate_decision": decision,
        "flow_on": flow_on,
        "is_act": act,
        "is_observe": observe,
        "is_hold": hold,
        "is_wait": wait,
        "any_act_allowed": any_act_allow,
    }


def wait_deadline(now: torch.Tensor, horizon_s: float = 1.0) -> torch.Tensor:
    return now + horizon_s
