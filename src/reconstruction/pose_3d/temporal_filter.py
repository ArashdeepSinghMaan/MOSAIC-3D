"""Roadmap step 5 — temporal filtering of 3D joints (One-Euro and per-joint Kalman)."""
from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """Speed-adaptive low-pass filter; NaN inputs keep the previous estimate."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-6)
        x = np.where(np.isnan(x), self.x_prev, x)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * np.nan_to_num(self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = np.where(np.isnan(self.x_prev), x, a * x + (1 - a) * self.x_prev)
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class KalmanJointFilter:
    """Independent constant-velocity Kalman filter per joint coordinate."""

    def __init__(self, num_joints: int, dt: float = 1 / 30, q: float = 1e-3, r: float = 1e-4):
        self.J, self.dt, self.q, self.r = num_joints, dt, q, r
        self.x = None
        self.P = None

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float).reshape(self.J, 3)
        F = np.array([[1, self.dt], [0, 1]]); H = np.array([[1.0, 0.0]])
        Q, R = self.q * np.eye(2), np.array([[self.r]])
        if self.x is None:
            self.x = np.stack([np.nan_to_num(z), np.zeros_like(z)], -1)  # (J,3,2)
            self.P = np.broadcast_to(np.eye(2), (self.J, 3, 2, 2)).copy()
            return z
        out = np.zeros_like(z)
        for j in range(self.J):
            for k in range(3):
                x, P = F @ self.x[j, k], F @ self.P[j, k] @ F.T + Q
                if not np.isnan(z[j, k]):
                    S = H @ P @ H.T + R
                    K = P @ H.T / S
                    x = x + (K * (z[j, k] - H @ x)).ravel()
                    P = (np.eye(2) - K @ H) @ P
                self.x[j, k], self.P[j, k] = x, P
                out[j, k] = x[0]
        return out
