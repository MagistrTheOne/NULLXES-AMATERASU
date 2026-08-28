from __future__ import annotations

from torch import nn


def tubelet_stem(in_ch: int, out_ch: int, tt: int, th: int, tw: int) -> nn.Conv3d:
    return nn.Conv3d(in_ch, out_ch, kernel_size=(tt, th, tw), stride=(tt, th, tw), bias=False)
