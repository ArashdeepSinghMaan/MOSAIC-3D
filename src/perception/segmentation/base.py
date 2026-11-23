from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.types import BBox


class BaseSegmenter(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, boxes: list[BBox]) -> list[np.ndarray]:
        """One HxW boolean mask per input box."""
