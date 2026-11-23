"""Learned HOI: edge classifier over the interaction graph (PyTorch Geometric)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .geometric import INTERACTION_LABELS


class EdgeHOIClassifier(nn.Module):
    def __init__(self, node_dim: int = 6, edge_dim: int = 8, hidden: int = 128,
                 num_classes: int = len(INTERACTION_LABELS), layers: int = 3):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.inp = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=edge_dim)
                                    for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, num_classes))

    def forward(self, data):
        h = self.inp(data.x)
        for conv in self.convs:
            h = h + torch.relu(conv(h, data.edge_index, data.edge_attr))
        s, d = data.edge_index
        return self.head(torch.cat([h[s], h[d], data.edge_attr], -1))   # (E, C)
