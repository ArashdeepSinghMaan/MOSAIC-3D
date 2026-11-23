"""Roadmap step 4 — nonlinear reprojection optimisation:
    X* = argmin_X  sum_i w_i * rho(|| p_i - pi(P_i X) ||^2)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def refine_nonlinear(P_list, pts, X0, weights=None, loss: str = "huber", f_scale: float = 5.0
                     ) -> np.ndarray:
    pts = np.asarray(pts, float)
    w = np.sqrt(np.ones(len(P_list)) if weights is None else np.asarray(weights, float))
    Ps = np.stack(P_list)

    def residuals(X):
        x = Ps @ np.r_[X, 1.0]
        uv = x[:, :2] / x[:, 2:3]
        return ((uv - pts) * w[:, None]).ravel()

    return least_squares(residuals, np.asarray(X0, float), loss=loss, f_scale=f_scale).x
