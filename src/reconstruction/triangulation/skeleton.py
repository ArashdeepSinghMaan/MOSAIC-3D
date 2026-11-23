"""Triangulate a full skeleton (J joints) from V views."""
from __future__ import annotations

import numpy as np

from .nonlinear import refine_nonlinear
from .ransac import triangulate_ransac
from .weighted import triangulate_weighted
from .linear import triangulate_dlt


def triangulate_skeleton(P_list: list[np.ndarray], kps: np.ndarray, conf: np.ndarray,
                         method: str = "ransac_nonlinear", conf_thresh: float = 0.3,
                         ransac_thresh_px: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """kps: (V,J,2), conf: (V,J). Returns X (J,3) with NaN for failed joints, and joint confidence (J,)."""
    V, J, _ = kps.shape
    X = np.full((J, 3), np.nan)
    jconf = np.zeros(J)
    for j in range(J):
        w = np.where(conf[:, j] >= conf_thresh, conf[:, j], 0.0)
        if (w > 0).sum() < 2:
            continue
        if method == "linear":
            m = w > 0
            Xj = triangulate_dlt([P for P, k in zip(P_list, m) if k], kps[m, j])
            inl = m
        elif method == "weighted":
            Xj, inl = triangulate_weighted(P_list, kps[:, j], w), w > 0
        else:
            Xj, inl = triangulate_ransac(P_list, kps[:, j], w, ransac_thresh_px)
            if Xj is not None and method.endswith("nonlinear"):
                Xj = refine_nonlinear([P for P, k in zip(P_list, inl) if k], kps[inl, j], Xj, w[inl])
        if Xj is not None:
            X[j] = Xj
            jconf[j] = float(w[inl].mean()) * min(1.0, inl.sum() / 3)
    return X, jconf


def reprojection_error(P_list, X: np.ndarray, kps: np.ndarray, conf: np.ndarray,
                       conf_thresh: float = 0.3) -> float:
    """Mean pixel error over visible (view, joint) pairs with a valid 3D joint."""
    errs = []
    for v, P in enumerate(P_list):
        for j in range(X.shape[0]):
            if np.isnan(X[j]).any() or conf[v, j] < conf_thresh:
                continue
            x = P @ np.r_[X[j], 1.0]
            errs.append(np.linalg.norm(x[:2] / x[2] - kps[v, j]))
    return float(np.mean(errs)) if errs else float("nan")
