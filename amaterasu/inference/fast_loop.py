from __future__ import annotations

from dataclasses import dataclass

import torch

from amaterasu.constants import H_CHUNK_MAX
from amaterasu.inference.clocks import ClockConfig
from amaterasu.model.agency.policy import select_intent
from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.model.flow.matching import FlowNCES
from amaterasu.model.vision.cache import VisualCache
from amaterasu.tensors.modality import IntentKind
from amaterasu.tensors.sample import AMATERASUBatch


@dataclass
class FastTickResult:
    hidden: torch.Tensor
    ssm_state: list[torch.Tensor] | None
    visual_cache: VisualCache | None
    nces_traj: torch.Tensor | None
    chosen_kind: torch.Tensor | None
    flow_on: torch.Tensor | None


def fast_tick(
    model: Amaterasu32B,
    batch: AMATERASUBatch,
    visual_cache: VisualCache | None,
    ssm_state: list[torch.Tensor] | None,
    chosen_kind: torch.Tensor | None,
    clocks: ClockConfig,
    tick: int = 0,
    episode_reset: torch.Tensor | None = None,
) -> FastTickResult:
    kind = int(chosen_kind[0].item()) if chosen_kind is not None else int(IntentKind.OBSERVE)
    if kind < 8:
        mode = "FAST_ACT"
    elif kind == int(IntentKind.HOLD):
        mode = "FAST_HOLD"
    elif kind == int(IntentKind.WAIT):
        mode = "FAST_WAIT"
    else:
        mode = "FAST_OBSERVE"
    nces_traj = batch.nces_traj
    horizon_mask = batch.horizon_mask
    if mode == "FAST_HOLD" and batch.nces_feat is not None:
        hm = horizon_mask if horizon_mask is not None else torch.ones(
            batch.nces_feat.shape[0], H_CHUNK_MAX, dtype=torch.bool, device=batch.nces_feat.device
        )
        nces_traj = FlowNCES.hold_trajectory(batch.nces_feat, int(hm.shape[1]), hm)
        horizon_mask = hm
    flow_t = torch.rand(batch.nces_feat.shape[0], device=batch.nces_feat.device) if mode in {"FAST_ACT", "FAST_HOLD"} else None
    out = model.forward_mode(
        mode,
        nces_feat=batch.nces_feat,
        nces_valid=batch.nces_valid,
        ecd_raw=batch.ecd_raw,
        ecd_topo=batch.ecd_topo,
        tactile=batch.tactile,
        tactile_valid=batch.tactile_valid,
        visual_cache=visual_cache,
        ssm_state=ssm_state,
        nces_traj=nces_traj if mode in {"FAST_ACT", "FAST_HOLD"} else None,
        horizon_mask=horizon_mask,
        node_mask=batch.node_mask if batch.node_mask is not None else batch.nces_valid,
        flow_t=flow_t,
        episode_reset=episode_reset,
        tick=tick,
    )
    return FastTickResult(
        hidden=out["hidden"],  # type: ignore[arg-type]
        ssm_state=out["ssm_state"],  # type: ignore[arg-type]
        visual_cache=out["visual_cache"],  # type: ignore[arg-type]
        nces_traj=out.get("nces_traj"),  # type: ignore[arg-type]
        chosen_kind=chosen_kind,
        flow_on=torch.tensor([mode in {"FAST_ACT", "FAST_HOLD"}], device=batch.nces_feat.device),
    )


def sensor_refresh(
    model: Amaterasu32B,
    batch: AMATERASUBatch,
    visual_cache: VisualCache | None,
) -> VisualCache:
    out = model.forward_mode(
        "FAST_SENSOR_REFRESH",
        nces_feat=batch.nces_feat,
        nces_valid=batch.nces_valid,
        ecd_raw=batch.ecd_raw,
        ecd_topo=batch.ecd_topo,
        video=batch.video,
        camera_valid_mask=batch.camera_valid_mask,
        frame_times=batch.frame_times,
        audio_mel=batch.audio_mel,
        audio_mask=batch.audio_mask,
        visual_cache=visual_cache,
    )
    cache = out["visual_cache"]
    if not isinstance(cache, VisualCache):
        raise RuntimeError("sensor refresh did not return a VisualCache")
    return cache
