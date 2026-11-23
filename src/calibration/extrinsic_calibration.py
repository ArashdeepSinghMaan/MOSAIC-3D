"""Extrinsic calibration: place every camera in a common world frame.

Strategy: a calibration board seen simultaneously by cameras defines the world frame
(board pose in a reference view), then each camera's pose follows from solvePnP.
Validation is by multi-view reprojection error.
"""
from __future__ import annotations

import cv2
import numpy as np

from .coordinate_transforms import CameraModel


def solve_board_pose(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Board(world) -> camera pose (R, t) via PnP."""
    ok, rvec, tvec = cv2.solvePnP(obj_pts.astype(np.float64), img_pts.astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def calibrate_extrinsics_from_board(cameras: dict[str, CameraModel],
                                    observations: dict[str, tuple[np.ndarray, np.ndarray]]
                                    ) -> dict[str, CameraModel]:
    """observations[camera_id] = (object_points (N,3), image_points (N,2)) of the same board pose.

    The board coordinate frame becomes the world frame.
    """
    for cid, (obj, img) in observations.items():
        cam = cameras[cid]
        cam.R, cam.t = solve_board_pose(obj, img, cam.K, cam.dist)
    return cameras


def stereo_refine(cam_a: CameraModel, cam_b: CameraModel, obj_pts: list, img_a: list, img_b: list
                  ) -> CameraModel:
    """Refine cam_b relative to cam_a with cv2.stereoCalibrate (intrinsics fixed)."""
    flags = cv2.CALIB_FIX_INTRINSIC
    _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_pts, img_a, img_b, cam_a.K, cam_a.dist, cam_b.K, cam_b.dist, cam_a.image_size, flags=flags)
    cam_b.R = R @ cam_a.R
    cam_b.t = (R @ cam_a.t.reshape(3, 1) + T).ravel()
    return cam_b


def reprojection_errors(cameras: dict[str, CameraModel], X: np.ndarray,
                        observations: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean pixel reprojection error per camera of 3D points X (N,3) vs observed (N,2)."""
    return {cid: float(np.linalg.norm(cameras[cid].project(X, distort=True) - obs, axis=1).mean())
            for cid, obs in observations.items()}
