"""Runs detector + pose + hands + segmentation on one frame -> CameraObservation."""
from __future__ import annotations

import numpy as np

from src.core.types import CameraObservation


class PerceptionPipeline:
    def __init__(self, detector, pose_estimator=None, hand_estimator=None, segmenter=None):
        self.detector, self.pose, self.hands, self.seg = detector, pose_estimator, hand_estimator, segmenter

    def __call__(self, camera_id: str, frame_id: int, timestamp: float, image: np.ndarray
                 ) -> CameraObservation:
        dets = self.detector(image)
        persons = [d for d in dets if d.class_name == "person"]
        if self.pose and persons:
            for d, kp in zip(persons, self.pose(image, [p.bbox for p in persons])):
                d.keypoints = kp
        if self.seg and dets:
            for d, m in zip(dets, self.seg(image, [d.bbox for d in dets])):
                d.mask = m
        hands = self.hands(image) if self.hands else []
        return CameraObservation(camera_id, frame_id, timestamp, dets, hands)
