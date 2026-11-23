"""Roadmap step 1 — linear (DLT) triangulation."""
from __future__ import annotations

import cv2
import numpy as np


def triangulate_dlt(P_list: list[np.ndarray], pts: np.ndarray) -> np.ndarray:
    """P_list: V projection matrices (3,4); pts: (V,2) pixel observations -> X (3,)."""
    A = []
    for P, (u, v) in zip(P_list, np.asarray(pts, float)):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate_opencv_pair(P1, P2, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Two-view OpenCV triangulation for N points -> (N,3)."""
    Xh = cv2.triangulatePoints(P1, P2, pts1.T.astype(float), pts2.T.astype(float))
    return (Xh[:3] / Xh[3]).T
