from __future__ import annotations

import cv2
import numpy as np

from src.core.types import CameraObservation, Track
from src.perception.hand_pose.base import HAND21_SKELETON
from src.perception.human_pose.base import COCO17_SKELETON


def _color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_keypoints(img, xy, conf, skeleton, color, thr=0.3):
    for a, b in skeleton:
        if conf[a] > thr and conf[b] > thr:
            cv2.line(img, tuple(map(int, xy[a])), tuple(map(int, xy[b])), color, 2)
    for (x, y), c in zip(xy, conf):
        if c > thr:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)


def draw_observation(img: np.ndarray, obs: CameraObservation) -> np.ndarray:
    out = img.copy()
    for d in obs.detections:
        c = _color(d.local_track_id or hash(d.class_name) % 1000)
        x1, y1, x2, y2 = map(int, d.bbox.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f"{d.class_name} {d.confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, c, 1)
        if d.keypoints is not None:
            draw_keypoints(out, d.keypoints.xy, d.keypoints.confidence, COCO17_SKELETON, c)
    for h in obs.hands:
        if h.keypoints is not None:
            draw_keypoints(out, h.keypoints.xy, h.keypoints.confidence, HAND21_SKELETON, (0, 255, 255))
    return out


def draw_tracks(img: np.ndarray, tracks: list[Track], global_ids: dict[int, int] | None = None) -> np.ndarray:
    out = img.copy()
    for t in tracks:
        gid = (global_ids or {}).get(t.local_track_id)
        c = _color(gid if gid is not None else t.local_track_id)
        x1, y1, x2, y2 = map(int, t.bbox.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"L{t.local_track_id}" + (f" G{gid}" if gid is not None else "")
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, .6, c, 2)
        for p, q in zip(t.history[-30:-1], t.history[-29:]):
            cv2.line(out, tuple(map(int, p)), tuple(map(int, q)), c, 1)
    return out


def mosaic(images: list[np.ndarray], cols: int = 2, width: int = 640) -> np.ndarray:
    tiles = [cv2.resize(i, (width, int(i.shape[0] * width / i.shape[1]))) for i in images]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
