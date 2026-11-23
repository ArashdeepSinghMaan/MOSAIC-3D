from __future__ import annotations

import numpy as np


def mpjpe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean per-joint position error. pred/gt: (..., J, 3)."""
    err = np.linalg.norm(pred - gt, axis=-1)
    valid = ~np.isnan(err) if mask is None else (mask & ~np.isnan(err))
    return float(err[valid].mean()) if valid.any() else float("nan")


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-transform pred (J,3) onto gt (J,3)."""
    mp, mg = pred.mean(0), gt.mean(0)
    p, g = pred - mp, gt - mg
    U, S, Vt = np.linalg.svd(p.T @ g)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (p ** 2).sum()
    return s * p @ R + mg


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = np.atleast_3d(pred).reshape(-1, *pred.shape[-2:]), gt.reshape(-1, *gt.shape[-2:])
    errs = []
    for p, g in zip(pred, gt):
        m = ~(np.isnan(p).any(1) | np.isnan(g).any(1))
        if m.sum() >= 3:
            errs.append(np.linalg.norm(procrustes_align(p[m], g[m]) - g[m], axis=1).mean())
    return float(np.mean(errs)) if errs else float("nan")


def pck(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """Percentage of correct keypoints within an absolute threshold (px for 2D, m for 3D)."""
    err = np.linalg.norm(pred - gt, axis=-1)
    v = ~np.isnan(err)
    return float((err[v] <= threshold).mean()) if v.any() else float("nan")


COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                        1.07, 1.07, .87, .87, .89, .89]) / 10.0


def oks(pred: np.ndarray, gt: np.ndarray, visible: np.ndarray, area: float,
        sigmas: np.ndarray = COCO_SIGMAS) -> float:
    d2 = ((pred - gt) ** 2).sum(-1)
    k2 = (2 * sigmas) ** 2
    e = d2 / (2 * area * k2 + np.spacing(1))
    return float(np.exp(-e)[visible > 0].mean()) if (visible > 0).any() else 0.0
