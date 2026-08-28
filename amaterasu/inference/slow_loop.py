from __future__ import annotations

from dataclasses import dataclass

import torch

from amaterasu.model.agency.policy import select_intent
from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.model.vision.cache import VisualCache
from amaterasu.tensors.sample import AMATERASUBatch


@dataclass
class SlowTickResult:
    hidden: torch.Tensor
    chosen_kind: torch.Tensor
    gate_decision: torch.Tensor
    flow_on: torch.Tensor
    eac: dict
    z: object
    visual_cache: VisualCache | None


def slow_tick(
    model: Amaterasu32B,
    batch: AMATERASUBatch,
    visual_cache: VisualCache | None,
    ssm_state: list[torch.Tensor] | None = None,
) -> SlowTickResult:
    out = model.forward_mode(
        "SLOW_AGENCY",
        nces_feat=batch.nces_feat,
        nces_valid=batch.nces_valid,
        ecd_raw=batch.ecd_raw,
        ecd_topo=batch.ecd_topo,
        input_ids=batch.input_ids,
        lang_mask=batch.lang_mask,
        tactile=batch.tactile,
        tactile_valid=batch.tactile_valid,
        visual_cache=visual_cache,
        ssm_state=ssm_state,
        episode_reset=batch.episode_reset,
    )
    eac = out["eac"]
    assert isinstance(eac, dict)
    decision = select_intent(eac["intent_scores"], eac["gate_logits"])
    return SlowTickResult(
        hidden=out["hidden"],  # type: ignore[arg-type]
        chosen_kind=decision["chosen_kind"],
        gate_decision=decision["gate_decision"],
        flow_on=decision["flow_on"],
        eac=eac,
        z=out.get("z"),
        visual_cache=out["visual_cache"],  # type: ignore[arg-type]
    )
