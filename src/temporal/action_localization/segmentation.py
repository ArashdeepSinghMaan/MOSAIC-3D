"""Temporal action segmentation (MS-TCN style) + frame->segment conversion."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DilatedResidual(nn.Module):
    def __init__(self, d, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=d, dilation=d)
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return x + self.out(F.relu(self.conv(x)))


class _Stage(nn.Module):
    def __init__(self, layers, ch, din, nc):
        super().__init__()
        self.inp = nn.Conv1d(din, ch, 1)
        self.layers = nn.Sequential(*[_DilatedResidual(2 ** i, ch) for i in range(layers)])
        self.out = nn.Conv1d(ch, nc, 1)

    def forward(self, x):
        return self.out(self.layers(self.inp(x)))


class MSTCN(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, stages: int = 4, layers: int = 10, ch: int = 64):
        super().__init__()
        self.s1 = _Stage(layers, ch, in_dim, num_classes)
        self.rest = nn.ModuleList([_Stage(layers, ch, num_classes, num_classes) for _ in range(stages - 1)])

    def forward(self, x):  # (B, T, D) -> list of (B, C, T)
        out = self.s1(x.transpose(1, 2))
        outs = [out]
        for s in self.rest:
            out = s(F.softmax(out, 1))
            outs.append(out)
        return outs


def frame_labels_to_segments(labels: np.ndarray, fps: float, background: int = 0) -> list[dict]:
    segs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] != background:
                segs.append({"label": int(labels[start]), "start": start / fps, "end": i / fps})
            start = i
    return segs
