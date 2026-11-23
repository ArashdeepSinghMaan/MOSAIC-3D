from __future__ import annotations

import cv2
import numpy as np

from src.core.registry import REID_MODELS
from src.core.types import BBox
from .base import BaseReID


@REID_MODELS.register("osnet")
class OSNetReID(BaseReID):
    """OSNet via torchreid. Swap for a Re-ID transformer by registering another class."""

    def __init__(self, model_name: str = "osnet_x1_0", weights: str | None = None,
                 device: str = "cuda", input_size=(256, 128)):
        import torch
        import torchreid
        self.torch, self.device, self.size = torch, device, input_size
        self.model = torchreid.models.build_model(model_name, num_classes=1000, pretrained=weights is None)
        if weights:
            torchreid.utils.load_pretrained_weights(self.model, weights)
        self.model.eval().to(device)
        self.mean = np.array([0.485, 0.456, 0.406]); self.std = np.array([0.229, 0.224, 0.225])

    def __call__(self, image: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 512))
        h, w = image.shape[:2]
        crops = []
        for b in boxes:
            x1, y1, x2, y2 = np.clip(b.xyxy, 0, [w, h, w, h]).astype(int)
            crop = image[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1), ::-1]
            crop = cv2.resize(crop, self.size[::-1]).astype(np.float32) / 255.0
            crops.append(((crop - self.mean) / self.std).transpose(2, 0, 1))
        with self.torch.no_grad():
            x = self.torch.tensor(np.stack(crops), dtype=self.torch.float32, device=self.device)
            f = self.model(x).cpu().numpy()
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
