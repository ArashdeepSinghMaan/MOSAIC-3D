"""Geometric hand-object features from consecutive SceneStates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import SceneState


@dataclass
class HandObjectFeatures:
    person_id: int
    object_id: int
    hand: str
    distance: float                 # m, nearest hand landmark (or wrist) to object centre
    approach_speed: float           # m/s, positive = getting closer
    relative_speed: float           # m/s, |v_hand - v_obj|
    object_speed: float             # m/s
    grip_aperture: float            # m, thumb tip <-> index tip (nan if no hand landmarks)


def _hand_points(person, side: str) -> np.ndarray | None:
    h = getattr(person, f"{side}_hand_3d")
    if h is not None and not np.isnan(h).all():
        return h
    wrist = person.pose_3d[9 if side == "left" else 10]
    return None if np.isnan(wrist).any() else wrist.reshape(1, 3)


def compute_hand_object_features(prev: SceneState | None, cur: SceneState) -> list[HandObjectFeatures]:
    feats = []
    dt = (cur.timestamp - prev.timestamp) if prev else None
    for pid, person in cur.persons.items():
        for side in ("left", "right"):
            hp = _hand_points(person, side)
            if hp is None:
                continue
            hc = np.nanmean(hp, 0)
            ap = np.linalg.norm(hp[4] - hp[8]) if hp.shape[0] == 21 else np.nan
            for oid, obj in cur.objects.items():
                d = float(np.nanmin(np.linalg.norm(hp - obj.position_3d, axis=1)))
                approach = rel = ospd = 0.0
                if prev and dt and dt > 0 and pid in prev.persons and oid in prev.objects:
                    php = _hand_points(prev.persons[pid], side)
                    if php is not None:
                        phc = np.nanmean(php, 0)
                        pd = float(np.nanmin(np.linalg.norm(php - prev.objects[oid].position_3d, axis=1)))
                        approach = (pd - d) / dt
                        vh = (hc - phc) / dt
                        vo = (obj.position_3d - prev.objects[oid].position_3d) / dt
                        rel, ospd = float(np.linalg.norm(vh - vo)), float(np.linalg.norm(vo))
                feats.append(HandObjectFeatures(pid, oid, side, d, approach, rel, ospd, float(ap)))
    return feats
