from __future__ import annotations

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.model.norms.rmsnorm import RMSNorm


class MambaLayer(nn.Module):
    """Selective SSM. A is per-channel [d_inner]; B,C are input-dependent of rank d_state."""

    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        d = cfg.d_model
        di = cfg.ssm_d_inner
        self.d_inner = di
        self.d_state = cfg.ssm_d_state
        self.dt_rank = cfg.ssm_dt_rank
        self.d_conv = cfg.ssm_d_conv
        self.in_proj = nn.Linear(d, 2 * di, bias=False)
        self.conv1d = nn.Conv1d(di, di, kernel_size=cfg.ssm_d_conv, groups=di, padding=cfg.ssm_d_conv - 1, bias=False)
        self.x_proj = nn.Linear(di, cfg.ssm_dt_rank + 2 * cfg.ssm_d_state, bias=False)
        self.dt_proj = nn.Linear(cfg.ssm_dt_rank, di, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, di + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d, bias=False)
        self.norm = RMSNorm(d, eps=cfg.rms_eps)

    def _scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        h0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, d_inner = u.shape
        n = self.d_state
        a = -torch.exp(self.A_log.float())
        h = u.new_zeros(batch, d_inner, n) if h0 is None else h0
        ys = []
        for t in range(seq):
            dt = delta[:, t]
            decay = torch.exp(dt.unsqueeze(-1) * a.view(1, d_inner, 1))
            dbu = u[:, t].unsqueeze(-1) * (dt.unsqueeze(-1) * b[:, t].unsqueeze(1))
            h = h * decay + dbu
            ys.append((h * c[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + u * self.D.to(dtype=u.dtype), h

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = u.transpose(1, 2)
        u = self.conv1d(u)[..., : x.shape[1]]
        u = u.transpose(1, 2)
        u = torch.nn.functional.silu(u)
        dbc = self.x_proj(u)
        dt, b, c = dbc.split((self.dt_rank, self.d_state, self.d_state), dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt))
        y, h = self._scan(u, dt, b, c, state)
        y = y * torch.nn.functional.silu(z)
        return residual + self.out_proj(y), h


class PhysicalSSM(nn.Module):
    def __init__(self, cfg: Amaterasu32BConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MambaLayer(cfg) for _ in range(cfg.ssm_layers)])
        self.tbptt_fast_ticks = cfg.tbptt_fast_ticks
        self.n_layers = cfg.ssm_layers
        self.d_inner = cfg.ssm_d_inner
        self.d_state = cfg.ssm_d_state

    def zero_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        return [
            torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ]

    def reset(self, states: list[torch.Tensor] | None, episode_reset: torch.Tensor | None) -> list[torch.Tensor] | None:
        if states is None or episode_reset is None or not bool(episode_reset.any()):
            return states
        out = []
        for h in states:
            mask = episode_reset.view(-1, 1, 1).to(h.dtype)
            out.append(h * (1.0 - mask))
        return out

    def maybe_detach(self, states: list[torch.Tensor] | None, tick: int) -> list[torch.Tensor] | None:
        """TBPTT: detach recurrent state every tbptt_fast_ticks. Inference may continue beyond 16."""
        if states is None or not self.training:
            return states
        if tick > 0 and tick % self.tbptt_fast_ticks == 0:
            return [h.detach() for h in states]
        return states

    def forward(
        self,
        x: torch.Tensor,
        states: list[torch.Tensor] | None = None,
        episode_reset: torch.Tensor | None = None,
        tick: int = 0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        states = self.reset(states, episode_reset)
        states = self.maybe_detach(states, tick)
        new_states: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            st = None if states is None else states[i]
            x, h = layer(x, st)
            new_states.append(h)
        return x, new_states
