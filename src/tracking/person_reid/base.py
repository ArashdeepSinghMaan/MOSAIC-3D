from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.types import BBox


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


class BaseReID(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        """(N, D) L2-normalised appearance embeddings."""
