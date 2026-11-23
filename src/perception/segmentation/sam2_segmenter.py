"""Box-prompted SAM 2 segmentation (image predictor). Video predictor can be added for propagation."""
from __future__ import annotations

import numpy as np

from src.core.registry import SEGMENTERS
from src.core.types import BBox
from .base import BaseSegmenter


@SEGMENTERS.register("sam2")
class SAM2Segmenter(BaseSegmenter):
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-base-plus", device: str = "cuda"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    def __call__(self, image: np.ndarray, boxes: list[BBox]) -> list[np.ndarray]:
        if not boxes:
            return []
        self.predictor.set_image(image[..., ::-1])  # BGR -> RGB
        masks, _, _ = self.predictor.predict(box=np.stack([b.xyxy for b in boxes]),
                                             multimask_output=False)
        masks = masks.reshape(len(boxes), *image.shape[:2])
        return [m > 0.5 for m in masks]
