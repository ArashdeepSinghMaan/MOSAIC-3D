"""Spatial-Temporal Graph Convolutional Network (Yan et al., 2018) for 3D skeletons."""
import numpy as np
import torch
import torch.nn as nn

from src.core.registry import ACTION_MODELS
from src.perception.human_pose.base import COCO17_SKELETON


def coco17_adjacency(num_joints: int = 17, edges=COCO17_SKELETON) -> np.ndarray:
    A = np.eye(num_joints)
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    D = np.diag(1 / np.sqrt(A.sum(1)))
    return D @ A @ D


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, t_kernel=9, stride=1, dropout=0.1):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.edge_importance = nn.Parameter(torch.ones_like(self.A))
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, (t_kernel, 1), (stride, 1), (t_kernel // 2, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(dropout))
        self.res = (nn.Identity() if cin == cout and stride == 1 else
                    nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x: (B, C, T, V)
        y = torch.einsum("bctv,vw->bctw", self.gcn(x), self.A * self.edge_importance)
        return self.relu(self.tcn(y) + self.res(x))


@ACTION_MODELS.register("stgcn")
class STGCN(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 3, num_joints: int = 17,
                 channels=(64, 64, 128, 128, 256, 256), A: np.ndarray | None = None):
        super().__init__()
        A = coco17_adjacency(num_joints) if A is None else A
        self.bn = nn.BatchNorm1d(in_ch * num_joints)
        layers, c = [], in_ch
        for i, co in enumerate(channels):
            layers.append(STGCNBlock(c, co, A, stride=2 if i in (2, 4) else 1))
            c = co
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):  # (B, T, J, C)
        B, T, V, C = x.shape
        x = self.bn(x.permute(0, 2, 3, 1).reshape(B, V * C, T)).reshape(B, V, C, T).permute(0, 2, 3, 1)
        return self.head(self.blocks(x).mean((2, 3)))
