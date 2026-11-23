from __future__ import annotations

import numpy as np

from src.core.registry import POSE_ESTIMATORS
from src.core.types import BBox, Keypoints2D
from .base import BasePoseEstimator


@POSE_ESTIMATORS.register("yolo_pose")
class YOLOPoseEstimator(BasePoseEstimator):
    """Bottom-up YOLO-Pose; predictions are matched back to the requested boxes by IoU."""

    def __init__(self, weights: str = "yolo11m-pose.pt", device: str = "cuda:0"):
        from ultralytics import YOLO
        self.model, self.device = YOLO(weights), device

    def __call__(self, image: np.ndarray, boxes: list[BBox]) -> list[Keypoints2D]:
        r = self.model.predict(image, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return [Keypoints2D(np.zeros((17, 2)), np.zeros(17)) for _ in boxes]
        pb = [BBox(*b) for b in r.boxes.xyxy.cpu().numpy()]
        xy, cf = r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
        out = []
        for b in boxes:
            i = int(np.argmax([b.iou(p) for p in pb]))
            out.append(Keypoints2D(xy[i].astype(np.float64), cf[i].astype(np.float64)))
        return out
