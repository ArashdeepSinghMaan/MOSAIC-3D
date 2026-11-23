"""Shared data contracts passed between MOSAIC-3D modules.

Every module boundary in the architecture (perception -> tracking -> association ->
reconstruction -> interaction -> temporal) exchanges these types, so models can be
swapped without touching downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BBox:
    """Axis-aligned box in pixel coordinates (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2])

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Keypoints2D:
    """K keypoints: xy (K,2) and per-joint confidence (K,)."""
    xy: np.ndarray
    confidence: np.ndarray
    skeleton: str = "coco17"  # coco17 | halpe26 | hand21 | ...


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: BBox
    mask: Optional[np.ndarray] = None          # HxW bool
    keypoints: Optional[Keypoints2D] = None
    embedding: Optional[np.ndarray] = None     # appearance / Re-ID feature
    local_track_id: Optional[int] = None


@dataclass
class CameraObservation:
    """Standardised per-camera output of the 2D perception layer."""
    camera_id: str
    frame_id: int
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    hands: list[Detection] = field(default_factory=list)

    def by_class(self, class_name: str) -> list[Detection]:
        return [d for d in self.detections if d.class_name == class_name]


@dataclass
class Track:
    """Single-camera track state."""
    camera_id: str
    local_track_id: int
    class_name: str
    bbox: BBox
    score: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: list[np.ndarray] = field(default_factory=list)
    detection: Optional[Detection] = None


@dataclass
class GlobalIdentity:
    """Cross-camera identity: global id -> {camera_id: local_track_id}."""
    global_id: int
    class_name: str
    local_ids: dict[str, int] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    last_position_3d: Optional[np.ndarray] = None
    last_seen: float = 0.0


@dataclass
class Person3D:
    global_id: int
    pose_3d: np.ndarray                       # (J,3) world coords, metres
    joint_confidence: np.ndarray              # (J,)
    left_hand_3d: Optional[np.ndarray] = None  # (21,3)
    right_hand_3d: Optional[np.ndarray] = None
    reprojection_error: Optional[float] = None


@dataclass
class Object3D:
    global_id: int
    class_name: str
    position_3d: np.ndarray                   # (3,)
    orientation: Optional[np.ndarray] = None  # (3,3) rotation, if 6-DoF available
    confidence: float = 1.0


@dataclass
class SceneState:
    """Everything known about the world at one timestamp."""
    timestamp: float
    persons: dict[int, Person3D] = field(default_factory=dict)
    objects: dict[int, Object3D] = field(default_factory=dict)
    interactions: list["InteractionEvent"] = field(default_factory=list)


@dataclass
class InteractionEvent:
    person_id: int
    object_id: int
    hand: str          # "left" | "right" | "body"
    label: str         # approaching | reaching | grasping | holding | moving | using | releasing | placing
    confidence: float
    timestamp: float
