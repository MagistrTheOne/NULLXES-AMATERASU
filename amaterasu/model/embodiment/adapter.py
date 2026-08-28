from __future__ import annotations

import torch
from torch import nn


class EmbodimentAdapter(nn.Module):
    """Outside the 31,740,290,560 universal count. Morphology AdaLN lives here, not in Z/HPT."""

    def __init__(self, d_nces_action: int, d_robot: int, d_cmd: int, ecd_d: int = 128) -> None:
        super().__init__()
        self.d_nces_action = d_nces_action
        self.state_in = nn.Linear(d_robot, d_nces_action, bias=False)
        self.cmd_out = nn.Linear(d_nces_action, d_cmd, bias=False)
        self.ecd_to_time = nn.Linear(ecd_d, 2 * d_nces_action, bias=False)

    def nparams(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        nces_desired: torch.Tensor,
        robot_state: torch.Tensor,
        ecd_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """nces_desired [B,H,N,D_action], robot_state [B,D_robot] → [B,H,N,D_cmd]."""
        fused = nces_desired + self.state_in(robot_state).view(robot_state.shape[0], 1, 1, -1)
        if ecd_raw is not None:
            scale, shift = self.ecd_to_time(ecd_raw).chunk(2, dim=-1)
            fused = fused * (1.0 + scale.view(-1, 1, 1, fused.shape[-1])) + shift.view(-1, 1, 1, fused.shape[-1])
        return self.cmd_out(fused)
