"""Per-person 3D skeleton reconstruction from a multi-view cluster of tracks."""
from __future__ import annotations

import numpy as np

from src.calibration.coordinate_transforms import CameraModel
from src.core.types import Person3D, Track
from src.reconstruction.triangulation.skeleton import reprojection_error, triangulate_skeleton
from .temporal_filter import OneEuroFilter


def bone_lengths(X: np.ndarray, skeleton: list[tuple[int, int]]) -> np.ndarray:
    return np.array([np.linalg.norm(X[a] - X[b]) for a, b in skeleton])


class Skeleton3DReconstructor:
    def __init__(self, cameras: dict[str, CameraModel], method: str = "ransac_nonlinear",
                 conf_thresh: float = 0.3, undistort: bool = True, smooth: bool = True,
                 num_joints: int = 17):
        self.cameras, self.method, self.conf_thresh = cameras, method, conf_thresh
        self.undistort, self.smooth, self.J = undistort, smooth, num_joints
        self.filters: dict[int, OneEuroFilter] = {}

    def __call__(self, global_id: int, cluster: dict[str, Track], timestamp: float) -> Person3D | None:
        views = [(cid, t) for cid, t in cluster.items()
                 if t.detection is not None and t.detection.keypoints is not None]
        if len(views) < 2:
            return None
        P_list, kps, conf = [], [], []
        for cid, t in views:
            cam = self.cameras[cid]
            xy = t.detection.keypoints.xy
            if self.undistort and np.any(cam.dist):
                xy = cam.undistort_points(xy)
            P_list.append(cam.P); kps.append(xy); conf.append(t.detection.keypoints.confidence)
        kps, conf = np.stack(kps), np.stack(conf)
        X, jc = triangulate_skeleton(P_list, kps, conf, self.method, self.conf_thresh)
        if self.smooth:
            f = self.filters.setdefault(global_id, OneEuroFilter())
            X = f(X, timestamp)
        err = reprojection_error(P_list, X, kps, conf, self.conf_thresh)
        return Person3D(global_id, X, jc, reprojection_error=err)
