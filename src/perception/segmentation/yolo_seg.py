from __future__ import annotations

import numpy as np

from src.core.registry import SEGMENTERS
from src.core.types import BBox
from .base import BaseSegmenter


@SEGMENTERS.register("yolo_seg")
class YOLOSegmenter(BaseSegmenter):
    def __init__(self, weights: str = "yolo11m-seg.pt", device: str = "cuda:0"):
        from ultralytics import YOLO
        self.model, self.device = YOLO(weights), device

    def __call__(self, image: np.ndarray, boxes: list[BBox]) -> list[np.ndarray]:
        r = self.model.predict(image, device=self.device, verbose=False, retina_masks=True)[0]
        if r.masks is None:
            return [np.zeros(image.shape[:2], bool) for _ in boxes]
        masks = r.masks.data.cpu().numpy() > 0.5
        pred = [BBox(*b) for b in r.boxes.xyxy.cpu().numpy()]
        out = []
        for b in boxes:  # match each requested box to best-IoU predicted mask
            ious = [b.iou(p) for p in pred]
            out.append(masks[int(np.argmax(ious))] if ious and max(ious) > 0.3
                       else np.zeros(image.shape[:2], bool))
        return out
