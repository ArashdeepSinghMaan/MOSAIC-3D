from __future__ import annotations

import numpy as np

from src.core.registry import DETECTORS
from src.core.types import BBox, Detection
from .base import BaseDetector


def ultralytics_to_detections(result) -> list[Detection]:
    out = []
    names = result.names
    for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy(),
                                           result.boxes.cls.cpu().numpy().astype(int)):
        out.append(Detection(names[cls], float(conf), BBox(x1, y1, x2, y2)))
    return out


@DETECTORS.register("yolo")
class YOLODetector(BaseDetector):
    def __init__(self, weights: str = "yolo11m.pt", device: str = "cuda:0", imgsz: int = 640,
                 half: bool = True, **kw):
        super().__init__(**kw)
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device, self.imgsz, self.half = device, imgsz, half

    def predict(self, image: np.ndarray) -> list[Detection]:
        r = self.model.predict(image, imgsz=self.imgsz, device=self.device, half=self.half,
                               conf=self.conf_threshold, verbose=False)[0]
        return ultralytics_to_detections(r)
