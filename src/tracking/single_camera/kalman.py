"""Constant-velocity Kalman filter on (cx, cy, aspect, height) — as in SORT/ByteTrack."""
from __future__ import annotations

import numpy as np


class BoxKalmanFilter:
    def __init__(self, std_pos: float = 1 / 20, std_vel: float = 1 / 160):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.sp, self.sv = std_pos, std_vel

    @staticmethod
    def xyxy_to_xyah(b):
        w, h = b[2] - b[0], b[3] - b[1]
        return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])

    @staticmethod
    def xyah_to_xyxy(m):
        cx, cy, a, h = m[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.sp * h, 2 * self.sp * h, 1e-2, 2 * self.sp * h,
               10 * self.sv * h, 10 * self.sv * h, 1e-5, 10 * self.sv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        q = np.diag(np.square([self.sp * h, self.sp * h, 1e-2, self.sp * h,
                               self.sv * h, self.sv * h, 1e-5, self.sv * h]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(self, mean, cov, z):
        h = mean[3]
        r = np.diag(np.square([self.sp * h, self.sp * h, 1e-1, self.sp * h]))
        S = self.H @ cov @ self.H.T + r
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.H @ mean)
        cov = (np.eye(8) - K @ self.H) @ cov
        return mean, cov
