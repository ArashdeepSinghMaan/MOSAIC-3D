"""Fuse 3D persons, hands and objects into a SceneState per timestamp, with history."""
from __future__ import annotations

from collections import deque

import numpy as np

from src.core.types import Object3D, Person3D, SceneState

COCO_LEFT_WRIST, COCO_RIGHT_WRIST = 9, 10


class SceneBuilder:
    def __init__(self, history: int = 900):
        self.history: deque[SceneState] = deque(maxlen=history)

    def build(self, timestamp: float, persons: list[Person3D], objects: list[Object3D],
              hands: dict[tuple[int, str], np.ndarray] | None = None) -> SceneState:
        scene = SceneState(timestamp, {p.global_id: p for p in persons},
                           {o.global_id: o for o in objects})
        for (pid, side), joints in (hands or {}).items():
            if pid in scene.persons:
                setattr(scene.persons[pid], f"{side}_hand_3d", joints)
        self.history.append(scene)
        return scene

    @staticmethod
    def person_wrists(persons: list[Person3D]) -> dict[int, dict[str, np.ndarray]]:
        return {p.global_id: {"left": p.pose_3d[COCO_LEFT_WRIST], "right": p.pose_3d[COCO_RIGHT_WRIST]}
                for p in persons}

    def trajectory(self, kind: str, gid: int) -> np.ndarray:
        pts = []
        for s in self.history:
            if kind == "object" and gid in s.objects:
                pts.append(s.objects[gid].position_3d)
            elif kind == "person" and gid in s.persons:
                pts.append(np.nanmean(s.persons[gid].pose_3d, 0))
        return np.array(pts).reshape(-1, 3)

    @staticmethod
    def to_json(scene: SceneState) -> dict:
        def arr(a):
            return None if a is None else np.where(np.isnan(a), None, np.round(a, 4)).tolist()
        return {"timestamp": scene.timestamp,
                "persons": {gid: {"pose_3d": arr(p.pose_3d), "left_hand_3d": arr(p.left_hand_3d),
                                  "right_hand_3d": arr(p.right_hand_3d),
                                  "reprojection_error": p.reprojection_error}
                            for gid, p in scene.persons.items()},
                "objects": {gid: {"class": o.class_name, "position_3d": arr(o.position_3d),
                                  "orientation": arr(o.orientation)} for gid, o in scene.objects.items()},
                "interactions": [vars(e) for e in scene.interactions]}
