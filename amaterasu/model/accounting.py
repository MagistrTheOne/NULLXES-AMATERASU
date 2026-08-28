from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from amaterasu.constants import (
    FROZEN_FAST_ACT_HOLD,
    FROZEN_FAST_ALWAYS,
    FROZEN_FAST_SENSOR_REFRESH_BOTH,
    FROZEN_FAST_SENSOR_REFRESH_VISION,
    FROZEN_FAST_STATE_ACTIVE,
    FROZEN_FAST_STATE_FLOW,
    FROZEN_FLOW,
    FROZEN_LM_HEAD,
    FROZEN_SLOW_ACT_HOLD,
    FROZEN_SLOW_ACTIVE,
    FROZEN_TOTAL,
    LEDGER,
)
from amaterasu.model.amaterasu import Amaterasu32B
from amaterasu.model.experts.physical_moe import PhysicalMoE
from amaterasu.model.hpt.fast_layer import HPTFastLayer
from amaterasu.model.hpt.slow_layer import HPTSlowDenseLayer, HPTSlowMoELayer


@dataclass
class LedgerReport:
    total: int
    components: dict[str, int]
    graphs: dict[str, int]
    diffs: dict[str, int] = field(default_factory=dict)


def nparams(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _hpt_buckets(hpt: nn.Module) -> dict[str, int]:
    buckets = {
        "shared_hpt_attn_norm": 0,
        "vision_ffns": 0,
        "language_ffns": 0,
        "physical_dense_ffns": 0,
        "physical_experts": 0,
        "agency_ffns": 0,
    }
    layers: list[nn.Module] = []
    layers.extend(list(hpt.fast_layers))
    layers.extend(list(hpt.slow_dense_layers))
    layers.extend(list(hpt.slow_moe_layers))
    for layer in layers:
        buckets["shared_hpt_attn_norm"] += nparams(layer.attn) + nparams(layer.norm_attn)
        buckets["vision_ffns"] += nparams(layer.ffn_vision) + nparams(layer.norm_vision)
        if isinstance(layer, HPTFastLayer):
            buckets["physical_dense_ffns"] += nparams(layer.ffn_physical) + nparams(layer.norm_physical)
        elif isinstance(layer, HPTSlowDenseLayer):
            buckets["language_ffns"] += nparams(layer.ffn_language) + nparams(layer.norm_language)
            buckets["physical_dense_ffns"] += nparams(layer.ffn_physical) + nparams(layer.norm_physical)
            buckets["agency_ffns"] += nparams(layer.ffn_agency) + nparams(layer.norm_agency)
        elif isinstance(layer, HPTSlowMoELayer):
            buckets["language_ffns"] += nparams(layer.ffn_language) + nparams(layer.norm_language)
            buckets["physical_experts"] += nparams(layer.moe) + nparams(layer.norm_physical)
            buckets["agency_ffns"] += nparams(layer.ffn_agency) + nparams(layer.norm_agency)
    return buckets


def _unused_routed_expert_params(model: Amaterasu32B) -> int:
    unused = 0
    for layer in model.hpt.slow_moe_layers:
        moe: PhysicalMoE = layer.moe
        per_expert = nparams(moe.routed[0])
        unused += (moe.n_routed - moe.topk) * per_expert
    return unused


def account(model: Amaterasu32B) -> LedgerReport:
    adapter_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad and name.startswith("adapter."):
            adapter_params += p.numel()
    if adapter_params:
        raise RuntimeError("adapter parameters must not appear on Amaterasu32B")

    components = {
        "vision_encoder": nparams(model.vision),
        "embeddings": nparams(model.embeddings),
        "nces_encoder": nparams(model.nces),
        "audio_encoder": nparams(model.audio),
        "tactile": nparams(model.tactile),
        "ssm": nparams(model.ssm),
        "memory": nparams(model.memory),
        "latent_dynamics": nparams(model.dynamics),
        "eac_gcis": nparams(model.eac),
        "flow": nparams(model.flow),
        "ecd": nparams(model.ecd),
    }
    components.update(_hpt_buckets(model.hpt))
    total = sum(components.values())
    walked = nparams(model)
    if walked != total:
        raise RuntimeError(f"named-component sum {total} != walked {walked}")

    diffs = {k: components[k] - LEDGER[k] for k in LEDGER}
    lm_head = nparams(model.embeddings.lm_head)
    flow = nparams(model.flow)
    unused_experts = _unused_routed_expert_params(model)
    slow_active = total - lm_head - flow - unused_experts
    graphs = {
        "TOTAL": total,
        "SLOW_ACTIVE": slow_active,
        "SLOW_ACT_HOLD": slow_active + flow,
        "FAST_STATE_ACTIVE": (
            nparams(model.hpt.fast_layers)
            + nparams(model.nces)
            + nparams(model.tactile)
            + nparams(model.ssm)
            + nparams(model.memory.working)
            + nparams(model.ecd)
        ),
        "FAST_SENSOR_REFRESH_VISION": nparams(model.vision),
        "FAST_SENSOR_REFRESH_BOTH": nparams(model.vision) + nparams(model.audio),
        "FAST_ALWAYS": 0,
        "FAST_ACT_HOLD": 0,
        "FAST_STATE_FLOW": 0,
        "LM_HEAD": lm_head,
        "FLOW": flow,
    }
    graphs["FAST_ALWAYS"] = graphs["FAST_STATE_ACTIVE"] + graphs["FAST_SENSOR_REFRESH_BOTH"]
    graphs["FAST_ACT_HOLD"] = graphs["FAST_ALWAYS"] + flow
    graphs["FAST_STATE_FLOW"] = graphs["FAST_STATE_ACTIVE"] + flow
    return LedgerReport(total=total, components=components, graphs=graphs, diffs=diffs)


def assert_frozen_total(report: LedgerReport) -> None:
    expected_graphs = {
        "TOTAL": FROZEN_TOTAL,
        "SLOW_ACTIVE": FROZEN_SLOW_ACTIVE,
        "SLOW_ACT_HOLD": FROZEN_SLOW_ACT_HOLD,
        "FAST_STATE_ACTIVE": FROZEN_FAST_STATE_ACTIVE,
        "FAST_SENSOR_REFRESH_VISION": FROZEN_FAST_SENSOR_REFRESH_VISION,
        "FAST_SENSOR_REFRESH_BOTH": FROZEN_FAST_SENSOR_REFRESH_BOTH,
        "FAST_ALWAYS": FROZEN_FAST_ALWAYS,
        "FAST_ACT_HOLD": FROZEN_FAST_ACT_HOLD,
        "FAST_STATE_FLOW": FROZEN_FAST_STATE_FLOW,
        "LM_HEAD": FROZEN_LM_HEAD,
        "FLOW": FROZEN_FLOW,
    }
    failures: list[str] = []
    if report.total != FROZEN_TOTAL:
        failures.append(f"TOTAL {report.total} != {FROZEN_TOTAL}")
    for k, exp in LEDGER.items():
        got = report.components[k]
        if got != exp:
            failures.append(f"{k}: {got} != {exp} (diff {got - exp})")
    for k, exp in expected_graphs.items():
        got = report.graphs[k]
        if got != exp:
            failures.append(f"graph {k}: {got} != {exp} (diff {got - exp})")
    if failures:
        raise SystemExit("PHASE II FAILS\n" + "\n".join(failures))
