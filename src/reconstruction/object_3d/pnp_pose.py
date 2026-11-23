"""Level 2 — 6-DoF pose of a known rigid object from 2D-3D correspondences (PnP + RANSAC)."""
from __future__ import annotations

import cv2
import numpy as np

from src.calibration.coordinate_transforms import CameraModel


def estimate_pose_pnp(model_points: np.ndarray, image_points: np.ndarray, cam: CameraModel,
                      reproj_thresh: float = 8.0) -> np.ndarray | None:
    """Returns 4x4 object->world transform, or None."""
    if len(model_points) < 4:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(model_points.astype(np.float64),
                                             image_points.astype(np.float64), cam.K, cam.dist,
                                             reprojectionError=reproj_thresh,
                                             flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 4:
        return None
    R_oc, _ = cv2.Rodrigues(rvec)                       # object -> camera
    T_oc = np.eye(4); T_oc[:3, :3], T_oc[:3, 3] = R_oc, tvec.ravel()
    T_wc = np.eye(4); T_wc[:3, :3], T_wc[:3, 3] = cam.R, cam.t.ravel()   # world -> camera
    return np.linalg.inv(T_wc) @ T_oc                   # object -> world


def foundationpose_stub(*_, **__):
    """Placeholder for FoundationPose integration (model-based 6D pose + tracking)."""
    raise NotImplementedError("Integrate NVlabs/FoundationPose here (Phase 4, Level 2).")
