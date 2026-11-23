"""3D constant-velocity Kalman tracking of object centres in the world frame."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class _Obj3DTrack:
    global_id: int
    class_name: str
    x: np.ndarray                 # (6,) pos+vel
    P: np.ndarray
    misses: int = 0
    trajectory: list[np.ndarray] = field(default_factory=list)


class ObjectTracker3D:
    def __init__(self, dt: float = 1 / 30, gate: float = 0.3, max_misses: int = 30,
                 q: float = 1e-2, r: float = 1e-3):
        self.dt, self.gate, self.max_misses = dt, gate, max_misses
        self.F = np.eye(6); self.F[:3, 3:] = dt * np.eye(3)
        self.H = np.eye(3, 6)
        self.Q, self.R = q * np.eye(6), r * np.eye(3)
        self.tracks: dict[int, _Obj3DTrack] = {}

    def step(self, observations: dict[int, tuple[str, np.ndarray]]) -> dict[int, np.ndarray]:
        """observations: global_id -> (class, xyz). Unknown ids spawn tracks. Returns filtered xyz."""
        for t in self.tracks.values():
            t.x = self.F @ t.x; t.P = self.F @ t.P @ self.F.T + self.Q
        for gid, (cls, z) in observations.items():
            z = np.asarray(z, float)
            if gid not in self.tracks:
                self.tracks[gid] = _Obj3DTrack(gid, cls, np.r_[z, np.zeros(3)], np.eye(6) * 0.1)
            t = self.tracks[gid]
            S = self.H @ t.P @ self.H.T + self.R
            K = t.P @ self.H.T @ np.linalg.inv(S)
            t.x = t.x + K @ (z - self.H @ t.x); t.P = (np.eye(6) - K @ self.H) @ t.P
            t.misses = 0
        for gid, t in list(self.tracks.items()):
            if gid not in observations:
                t.misses += 1
                if t.misses > self.max_misses:
                    del self.tracks[gid]; continue
            t.trajectory = (t.trajectory + [t.x[:3].copy()])[-1000:]
        return {gid: t.x[:3].copy() for gid, t in self.tracks.items()}

    def associate_anonymous(self, positions: np.ndarray) -> list[int | None]:
        """Match un-identified 3D points to existing tracks by Euclidean gate."""
        ids = list(self.tracks)
        if not ids or len(positions) == 0:
            return [None] * len(positions)
        pred = np.array([self.tracks[i].x[:3] for i in ids])
        cost = np.linalg.norm(positions[:, None] - pred[None], axis=2)
        r, c = linear_sum_assignment(cost)
        out: list[int | None] = [None] * len(positions)
        for i, j in zip(r, c):
            if cost[i, j] <= self.gate:
                out[i] = ids[j]
        return out
