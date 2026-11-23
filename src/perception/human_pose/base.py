from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.types import BBox, Keypoints2D

COCO17_JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                 "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                 "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
COCO17_SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
                   (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                   (3, 5), (4, 6)]


class BasePoseEstimator(ABC):
    """Top-down: given person boxes, return one Keypoints2D per box."""

    @abstractmethod
    def __call__(self, image: np.ndarray, boxes: list[BBox]) -> list[Keypoints2D]: ...
