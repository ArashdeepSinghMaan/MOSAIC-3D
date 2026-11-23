"""Intrinsic calibration from checkerboard or ChArUco images (OpenCV)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class IntrinsicResult:
    K: np.ndarray
    dist: np.ndarray
    rms_reprojection_error: float
    image_size: tuple[int, int]
    n_views: int


def checkerboard_object_points(pattern: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def calibrate_checkerboard(images: list[np.ndarray], pattern=(9, 6), square_size=0.025) -> IntrinsicResult:
    objp = checkerboard_object_points(pattern, square_size)
    obj_pts, img_pts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern, None)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            obj_pts.append(objp)
            img_pts.append(corners)
    if len(obj_pts) < 5:
        raise RuntimeError(f"Only {len(obj_pts)} valid checkerboard views; need >= 5")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return IntrinsicResult(K, dist.ravel(), float(rms), size, len(obj_pts))


def calibrate_charuco(images: list[np.ndarray], squares=(7, 5), square_len=0.04, marker_len=0.03,
                      dictionary: int = cv2.aruco.DICT_5X5_100) -> IntrinsicResult:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.CharucoBoard(squares, square_len, marker_len, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    obj_pts, img_pts, size = [], [], None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        size = gray.shape[::-1]
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is not None and len(ch_ids) >= 6:
            o, i = board.matchImagePoints(ch_corners, ch_ids)
            obj_pts.append(o)
            img_pts.append(i)
    if len(obj_pts) < 5:
        raise RuntimeError(f"Only {len(obj_pts)} valid ChArUco views; need >= 5")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return IntrinsicResult(K, dist.ravel(), float(rms), size, len(obj_pts))


def load_images(folder: str | Path, pattern: str = "*.png", every_n: int = 1) -> list[np.ndarray]:
    paths = sorted(Path(folder).glob(pattern))[::every_n]
    return [cv2.imread(str(p)) for p in paths]


def frames_from_video(video: str | Path, every_n: int = 15, max_frames: int = 80) -> list[np.ndarray]:
    cap, frames, i = cv2.VideoCapture(str(video)), [], 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames
