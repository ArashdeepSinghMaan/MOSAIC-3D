from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.types import Detection


class BaseDetector(ABC):
    def __init__(self, classes: list[str] | None = None, conf_threshold: float = 0.3):
        self.classes = set(classes) if classes else None
        self.conf_threshold = conf_threshold

    @abstractmethod
    def predict(self, image: np.ndarray) -> list[Detection]: ...

    def __call__(self, image: np.ndarray) -> list[Detection]:
        dets = self.predict(image)
        return [d for d in dets if d.confidence >= self.conf_threshold
                and (self.classes is None or d.class_name in self.classes)]
