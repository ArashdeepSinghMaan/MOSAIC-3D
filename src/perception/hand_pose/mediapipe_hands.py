from __future__ import annotations

import numpy as np

from src.core.registry import HAND_ESTIMATORS
from src.core.types import BBox, Detection, Keypoints2D
from .base import BaseHandEstimator


@HAND_ESTIMATORS.register("mediapipe")
class MediaPipeHandEstimator(BaseHandEstimator):
    def __init__(self, max_hands: int = 4, min_conf: float = 0.5):
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                                              min_detection_confidence=min_conf)

    def __call__(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        res = self.hands.process(image[..., ::-1])
        out = []
        if not res.multi_hand_landmarks:
            return out
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            xy = np.array([[p.x * w, p.y * h] for p in lm.landmark])
            score = handed.classification[0].score
            # MediaPipe labels assume a mirrored selfie view; flip for third-person cameras.
            label = "right_hand" if handed.classification[0].label == "Left" else "left_hand"
            x1, y1 = xy.min(0)
            x2, y2 = xy.max(0)
            out.append(Detection(label, float(score), BBox(x1, y1, x2, y2),
                                 keypoints=Keypoints2D(xy, np.full(21, score), "hand21")))
        return out
