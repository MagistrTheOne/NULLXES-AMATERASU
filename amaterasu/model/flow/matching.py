from __future__ import annotations

import math

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.constants import H_CHUNK_MAX, N_NODES_MAX
from amaterasu.model.transformer_block import PreNormBlock
from amaterasu.tensors.nces_schema import GRASP, POS, ROT6D

D_ACTION_PACK = 8


def pack_nces_action(traj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Parameter-free pack: per step, 64 nodes × 8 dims → 512. Invalid nodes zeroed."""
    pos = traj[..., POS]
    rot = traj[..., ROT6D][..., :3]
    grasp = traj[..., GRASP]
    packed8 = torch.cat([pos, rot, grasp], dim=-1)
    packed8 = packed8 * node_mask.unsqueeze(1).unsqueeze(-1).to(packed8.dtype)
    b, h, n, d8 = packed8.shape
    if n > N_NODES_MAX:
        raise ValueError("node count exceeds N_NODES_MAX")
    if n < N_NODES_MAX:
        pad = packed8.new_zeros(b, h, N_NODES_MAX - n, d8)
        packed8 = torch.cat([packed8, pad], dim=2)
    return packed8.reshape(b, h, N_NODES_MAX * D_ACTION_PACK)


def unpack_nces_action(packed: torch.Tensor, template: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Write the 8-d action slice back into a full NCES trajectory template."""
    b, h, _ = packed.shape
    x = packed.reshape(b, h, N_NODES_MAX, D_ACTION_PACK)
    n = template.shape[2]
    x = x[:, :, :n]
    out = template.clone()
    out[..., POS] = x[..., 0:3]
    rot = out[..., ROT6D]
    rot[..., :3] = x[..., 3:6]
    out[..., ROT6D] = rot
    out[..., GRASP] = x[..., 6:8]
    return out * node_mask.unsqueeze(1).unsqueeze(-1).to(out.dtype)


def fourier_time(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.arange(1, half + 1, device=t.device, dtype=torch.float32)
    ang = t.float().unsqueeze(-1) * freq * (2.0 * math.pi)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowBlock(nn.Module):
    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.block = PreNormBlock(
            cfg.flow_d,
            cfg.flow_heads,
            cfg.flow_kv,
            cfg.flow_d_head,
            cfg.flow_d_ff,
            rope_theta=cfg.rope_theta,
            eps=cfg.rms_eps,
        )
        self.adaln = nn.Linear(cfg.flow_time_d, 2 * cfg.flow_d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        position_ids: torch.Tensor,
        attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        scale, shift = self.adaln(t_emb).chunk(2, dim=-1)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.block(x, position_ids, attn_bias=attn_bias)


class FlowNCES(nn.Module):
    """12 × VE-width blocks + AdaLN + time MLP + 512↔2048 IO. Pack/unpack is parameter-free."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.h_max = H_CHUNK_MAX
        self.time_d = cfg.flow_time_d
        self.in_proj = nn.Linear(cfg.flow_io, cfg.flow_d, bias=False)
        self.out_proj = nn.Linear(cfg.flow_d, cfg.flow_io, bias=False)
        self.time_mlp_1 = nn.Linear(cfg.flow_time_d, cfg.flow_time_d, bias=False)
        self.time_mlp_2 = nn.Linear(cfg.flow_time_d, cfg.flow_time_d, bias=False)
        self.layers = nn.ModuleList([FlowBlock(cfg) for _ in range(cfg.flow_layers)])

    def vector_field(
        self,
        x512: torch.Tensor,
        horizon_mask: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in_proj(x512)
        t_emb = self.time_mlp_2(torch.nn.functional.silu(self.time_mlp_1(fourier_time(flow_t, self.time_d))))
        b, h, _ = x.shape
        pos = torch.arange(h, device=x.device).view(1, h).expand(b, h)
        key_invalid = ~horizon_mask.unsqueeze(1).unsqueeze(1)
        attn_bias = torch.zeros(b, 1, h, h, device=x.device, dtype=torch.float32)
        attn_bias = attn_bias.masked_fill(key_invalid, torch.finfo(torch.float32).min)
        for layer in self.layers:
            x = layer(x, t_emb, pos, attn_bias)
        pred = self.out_proj(x)
        return pred * horizon_mask.unsqueeze(-1).to(pred.dtype)

    def forward(
        self,
        nces_traj: torch.Tensor,
        node_mask: torch.Tensor,
        horizon_mask: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_nces_action(nces_traj, node_mask)
        pred512 = self.vector_field(packed, horizon_mask, flow_t)
        traj = unpack_nces_action(pred512, nces_traj, node_mask)
        return pred512, traj

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t = (1-t) x0 + t x1; target u = x1 - x0."""
        t_ = t.view(-1, 1, 1).to(x0.dtype)
        xt = (1.0 - t_) * x0 + t_ * x1
        return xt, x1 - x0

    def sample(
        self,
        x0: torch.Tensor,
        node_mask: torch.Tensor,
        horizon_mask: torch.Tensor,
        nfe: int,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """Euler integration from noise/current state to desired NCES. NFE is config, not a freeze."""
        x = pack_nces_action(x0, node_mask)
        dt = 1.0 / max(int(nfe), 1)
        for i in range(max(int(nfe), 1)):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=torch.float32)
            u = self.vector_field(x, horizon_mask, t)
            x = x + dt * u
        return unpack_nces_action(x, template, node_mask)

    @staticmethod
    def hold_trajectory(current: torch.Tensor, h: int, horizon_mask: torch.Tensor) -> torch.Tensor:
        """HOLD is a desired-state trajectory (repeat current NCES), never zero torque."""
        traj = current.unsqueeze(1).expand(-1, h, -1, -1).contiguous()
        return traj * horizon_mask.view(horizon_mask.shape[0], h, 1, 1).to(traj.dtype)
