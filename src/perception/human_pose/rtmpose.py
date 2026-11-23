from __future__ import annotations

import numpy as np

from src.core.registry import POSE_ESTIMATORS
from src.core.types import BBox, Keypoints2D
from .base import BasePoseEstimator


@POSE_ESTIMATORS.register("rtmpose")
class RTMPoseEstimator(BasePoseEstimator):
    """MMPose top-down inference (RTMPose / ViTPose share this wrapper — pass config+checkpoint)."""

    def __init__(self, config: str, checkpoint: str, device: str = "cuda:0"):
        from mmpose.apis import init_model
        self.model = init_model(config, checkpoint, device=device)

    def __call__(self, image: np.ndarray, boxes: list[BBox]) -> list[Keypoints2D]:
        from mmpose.apis import inference_topdown
        if not boxes:
            return []
        results = inference_topdown(self.model, image, np.stack([b.xyxy for b in boxes]))
        out = []
        for r in results:
            inst = r.pred_instances
            out.append(Keypoints2D(inst.keypoints[0].astype(np.float64),
                                   inst.keypoint_scores[0].astype(np.float64)))
        return out


POSE_ESTIMATORS.register("vitpose")(RTMPoseEstimator)
