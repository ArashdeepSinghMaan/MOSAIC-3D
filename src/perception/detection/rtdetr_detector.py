from __future__ import annotations

import numpy as np

from src.core.registry import DETECTORS
from src.core.types import Detection
from .base import BaseDetector
from .yolo_detector import ultralytics_to_detections


@DETECTORS.register("rtdetr")
class RTDETRDetector(BaseDetector):
    def __init__(self, weights: str = "rtdetr-l.pt", device: str = "cuda:0", imgsz: int = 640, **kw):
        super().__init__(**kw)
        from ultralytics import RTDETR
        self.model = RTDETR(weights)
        self.device, self.imgsz = device, imgsz

    def predict(self, image: np.ndarray) -> list[Detection]:
        r = self.model.predict(image, imgsz=self.imgsz, device=self.device,
                               conf=self.conf_threshold, verbose=False)[0]
        return ultralytics_to_detections(r)
