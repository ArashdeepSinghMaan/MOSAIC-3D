"""Roadmap step 3 — outlier rejection: RANSAC over view pairs, refit on inliers."""
from __future__ import annotations

import itertools

import numpy as np

from .weighted import triangulate_weighted


def _reproj(P, X, uv):
    x = P @ np.r_[X, 1.0]
    return np.linalg.norm(x[:2] / x[2] - uv)


def triangulate_ransac(P_list, pts, weights=None, thresh_px: float = 15.0, min_views: int = 2
                       ) -> tuple[np.ndarray | None, np.ndarray]:
    """Returns (X or None, inlier mask over views)."""
    V = len(P_list)
    pts = np.asarray(pts, float)
    w = np.ones(V) if weights is None else np.asarray(weights, float)
    valid = w > 1e-3
    best_inl, best_score = np.zeros(V, bool), -1.0
    for i, j in itertools.combinations(np.where(valid)[0], 2):
        X = triangulate_weighted([P_list[i], P_list[j]], pts[[i, j]], w[[i, j]])
        if X is None:
            continue
        # cheirality: point must be in front of both cameras
        if any((P_list[k] @ np.r_[X, 1])[2] <= 0 for k in (i, j)):
            continue
        inl = np.array([valid[k] and _reproj(P_list[k], X, pts[k]) < thresh_px for k in range(V)])
        score = w[inl].sum()
        if score > best_score:
            best_score, best_inl = score, inl
    if best_inl.sum() < min_views:
        return None, best_inl
    X = triangulate_weighted([P for P, k in zip(P_list, best_inl) if k], pts[best_inl], w[best_inl])
    return X, best_inl
