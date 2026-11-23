"""Roadmap step 2 — confidence-weighted DLT (w_{i,j} scales each view's equations)."""
from __future__ import annotations

import numpy as np


def triangulate_weighted(P_list: list[np.ndarray], pts: np.ndarray, weights: np.ndarray,
                         min_views: int = 2, min_weight: float = 1e-3) -> np.ndarray | None:
    pts, w = np.asarray(pts, float), np.asarray(weights, float)
    keep = w > min_weight
    if keep.sum() < min_views:
        return None
    A = []
    for P, (u, v), wi in zip([P for P, k in zip(P_list, keep) if k], pts[keep], w[keep]):
        r1, r2 = u * P[2] - P[0], v * P[2] - P[1]
        A.append(wi * r1 / (np.linalg.norm(r1) + 1e-12))
        A.append(wi * r2 / (np.linalg.norm(r2) + 1e-12))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]
