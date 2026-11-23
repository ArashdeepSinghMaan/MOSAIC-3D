"""Synthetic camera rigs and skeletons for geometry tests."""
import numpy as np

from src.calibration.coordinate_transforms import CameraModel


def look_at(center, target=(0, 0, 1.0), up=(0, 0, 1)):
    c, t, u = map(np.asarray, (center, target, up))
    z = (t - c) / np.linalg.norm(t - c)
    x = np.cross(z, u); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ c


def make_rig(n: int = 3, radius: float = 4.0, height: float = 2.0) -> dict[str, CameraModel]:
    K = np.array([[1400.0, 0, 960], [0, 1400, 540], [0, 0, 1]])
    cams = {}
    for i in range(n):
        a = 2 * np.pi * i / n
        R, t = look_at([radius * np.cos(a), radius * np.sin(a), height])
        cams[f"cam_{i + 1:02d}"] = CameraModel(f"cam_{i + 1:02d}", K.copy(), np.zeros(5), R, t)
    return cams


def random_skeleton(J: int = 17, seed: int = 0, center=(0, 0, 1.0)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-0.4, 0.4, (J, 3)) * [1, 1, 2]


def project_skeleton(cams, X, noise_px: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    kps = np.stack([c.project(X) for c in cams.values()]) + rng.normal(0, noise_px, (len(cams), len(X), 2))
    return kps, np.ones(kps.shape[:2])
