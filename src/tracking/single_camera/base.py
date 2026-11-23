from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import Detection, Track


class BaseTracker(ABC):
    """Per-camera tracker. Local IDs are only unique within one camera."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]: ...

    @abstractmethod
    def reset(self) -> None: ...
