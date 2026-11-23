"""ByteTrack: two-stage association (high-score then low-score detections) with Kalman motion."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.registry import TRACKERS
from src.core.types import BBox, Detection, Track
from .base import BaseTracker
from .kalman import BoxKalmanFilter


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def linear_assignment(cost: np.ndarray, thresh: float):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    matches = [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]
    mr = {i for i, _ in matches}; mc = {j for _, j in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in mr], \
        [j for j in range(cost.shape[1]) if j not in mc]


class _KTrack:
    def __init__(self, tid, det: Detection, kf: BoxKalmanFilter):
        self.id, self.kf, self.det = tid, kf, det
        self.mean, self.cov = kf.initiate(kf.xyxy_to_xyah(det.bbox.xyxy))
        self.hits, self.age, self.tsu = 1, 0, 0
        self.history = [det.bbox.center]

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1; self.tsu += 1

    def update(self, det: Detection):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, self.kf.xyxy_to_xyah(det.bbox.xyxy))
        self.det, self.tsu = det, 0
        self.hits += 1
        self.history = (self.history + [det.bbox.center])[-300:]

    @property
    def xyxy(self):
        return self.kf.xyah_to_xyxy(self.mean)


@TRACKERS.register("bytetrack")
class ByteTracker(BaseTracker):
    def __init__(self, camera_id: str = "cam_00", high_thresh: float = 0.5, low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6, match_thresh: float = 0.8, max_age: int = 30,
                 min_hits: int = 2, class_name: str | None = None):
        super().__init__(camera_id)
        self.high, self.low, self.new_t = high_thresh, low_thresh, new_track_thresh
        self.match_thresh, self.max_age, self.min_hits = match_thresh, max_age, min_hits
        self.class_name = class_name
        self.kf = BoxKalmanFilter()
        self.reset()

    def reset(self):
        self.tracks: list[_KTrack] = []
        self._next_id = 1

    def _match(self, tracks, dets, thresh):
        cost = 1 - iou_matrix(np.array([t.xyxy for t in tracks]).reshape(-1, 4),
                              np.array([d.bbox.xyxy for d in dets]).reshape(-1, 4))
        return linear_assignment(cost, thresh)

    def update(self, detections: list[Detection]) -> list[Track]:
        if self.class_name:
            detections = [d for d in detections if d.class_name == self.class_name]
        for t in self.tracks:
            t.predict()
        high = [d for d in detections if d.confidence >= self.high]
        low = [d for d in detections if self.low <= d.confidence < self.high]

        m1, ut, ud = self._match(self.tracks, high, self.match_thresh)
        for i, j in m1:
            self.tracks[i].update(high[j])
        remaining = [self.tracks[i] for i in ut]
        m2, ut2, _ = self._match(remaining, low, 0.5)
        for i, j in m2:
            remaining[i].update(low[j])
        for j in ud:
            if high[j].confidence >= self.new_t:
                self.tracks.append(_KTrack(self._next_id, high[j], self.kf))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.tsu <= self.max_age]

        out = []
        for t in self.tracks:
            if t.tsu == 0 and (t.hits >= self.min_hits or t.age <= self.min_hits):
                t.det.local_track_id = t.id
                out.append(Track(self.camera_id, t.id, t.det.class_name, BBox(*t.xyxy),
                                 t.det.confidence, t.age, t.hits, t.tsu, list(t.history), t.det))
        return out
