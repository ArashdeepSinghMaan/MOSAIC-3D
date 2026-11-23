"""Camera model and coordinate transforms.  Convention: X_cam = R @ X_world + t."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class CameraModel:
    camera_id: str
    K: np.ndarray                                   # (3,3)
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))   # world -> camera
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    image_size: tuple[int, int] = (1920, 1080)      # (w, h)

    @property
    def P(self) -> np.ndarray:
        return projection_matrix(self.K, self.R, self.t)

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t.reshape(3)

    def project(self, X: np.ndarray, distort: bool = False) -> np.ndarray:
        return project_points(X, self.K, self.R, self.t, self.dist if distort else None)

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Distorted pixels -> undistorted pixels (same K)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    # ---- I/O (configs/cameras/*.yaml & datasets/**/calibration/*.yaml) ----
    def to_dict(self) -> dict:
        return {"camera_id": self.camera_id, "image_size": list(self.image_size),
                "K": self.K.tolist(), "dist": np.asarray(self.dist).ravel().tolist(),
                "R": self.R.tolist(), "t": np.asarray(self.t).ravel().tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(d["camera_id"], np.array(d["K"], float), np.array(d.get("dist", [0] * 5), float),
                   np.array(d.get("R", np.eye(3)), float), np.array(d.get("t", [0, 0, 0]), float),
                   tuple(d.get("image_size", (1920, 1080))))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "CameraModel":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


def projection_matrix(K, R, t) -> np.ndarray:
    return K @ np.hstack([R, np.asarray(t, float).reshape(3, 1)])


def project_points(X, K, R, t, dist=None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    if dist is not None and np.any(dist):
        rvec, _ = cv2.Rodrigues(R)
        pts, _ = cv2.projectPoints(X, rvec, np.asarray(t, float), K, dist)
        return pts.reshape(-1, 2)
    Xc = X @ R.T + np.asarray(t).reshape(1, 3)
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3]


def invert_extrinsics(R, t) -> tuple[np.ndarray, np.ndarray]:
    """world->camera (R,t)  <->  camera->world (R', t')."""
    Rinv = R.T
    return Rinv, -Rinv @ np.asarray(t).reshape(3)


def to_homogeneous(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).reshape(3)
    return T


def world_to_camera(X, R, t) -> np.ndarray:
    return np.asarray(X).reshape(-1, 3) @ R.T + np.asarray(t).reshape(1, 3)


def camera_to_world(Xc, R, t) -> np.ndarray:
    return (np.asarray(Xc).reshape(-1, 3) - np.asarray(t).reshape(1, 3)) @ R


def relative_pose(cam_a: CameraModel, cam_b: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """Transform from camera-a frame to camera-b frame."""
    R = cam_b.R @ cam_a.R.T
    t = cam_b.t.reshape(3) - R @ cam_a.t.reshape(3)
    return R, t


def fundamental_matrix(cam_a: CameraModel, cam_b: CameraModel) -> np.ndarray:
    R, t = relative_pose(cam_a, cam_b)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    F = np.linalg.inv(cam_b.K).T @ E @ np.linalg.inv(cam_a.K)
    return F / (np.linalg.norm(F) + 1e-12)


def load_camera_rig(calib_dir: str | Path) -> dict[str, CameraModel]:
    return {c.camera_id: c for c in (CameraModel.load(p) for p in sorted(Path(calib_dir).glob("*.yaml")))}
