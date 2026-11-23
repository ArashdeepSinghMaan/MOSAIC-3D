"""Associate local tracks across cameras into multi-view clusters.

Affinity combines (a) symmetric epipolar distance of body keypoints / box centres,
(b) appearance cosine similarity, then clusters greedily so each cluster has at most
one track per camera. Kept separate from single-camera tracking by design.
"""
from __future__ import annotations

import itertools

import numpy as np

from src.calibration.coordinate_transforms import CameraModel
from src.core.types import Track


def epipolar_distance(F: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Symmetric point-to-epipolar-line distance (pixels) for matched points (N,2)."""
    pa_h = np.c_[pa, np.ones(len(pa))]; pb_h = np.c_[pb, np.ones(len(pb))]
    lb = pa_h @ F.T          # lines in image b
    la = pb_h @ F            # lines in image a
    db = np.abs(np.sum(lb * pb_h, 1)) / (np.linalg.norm(lb[:, :2], axis=1) + 1e-9)
    da = np.abs(np.sum(la * pa_h, 1)) / (np.linalg.norm(la[:, :2], axis=1) + 1e-9)
    return (da + db) / 2


def _track_points(t: Track, min_conf: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    d = t.detection
    if d is not None and d.keypoints is not None:
        return d.keypoints.xy, d.keypoints.confidence >= min_conf
    c = t.bbox.center.reshape(1, 2)
    return c, np.array([True])


class CrossCameraAssociator:
    def __init__(self, cameras: dict[str, CameraModel], epipolar_thresh_px: float = 30.0,
                 w_geometry: float = 0.7, w_appearance: float = 0.3, min_affinity: float = 0.3):
        self.cameras = cameras
        # Un-normalised F so epipolar distances are in pixels.
        self.F = {(a, b): self._unnormalised_F(a, b) for a, b in itertools.permutations(cameras, 2)}
        self.tau, self.wg, self.wa, self.min_aff = epipolar_thresh_px, w_geometry, w_appearance, min_affinity

    def _unnormalised_F(self, a, b):
        ca, cb = self.cameras[a], self.cameras[b]
        R = cb.R @ ca.R.T
        t = cb.t.reshape(3) - R @ ca.t.reshape(3)
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        return np.linalg.inv(cb.K).T @ (tx @ R) @ np.linalg.inv(ca.K)

    def affinity(self, ta: Track, tb: Track) -> float:
        pa, va = _track_points(ta); pb, vb = _track_points(tb)
        if len(pa) == len(pb) and len(pa) > 1:
            m = va & vb
            if m.sum() < 3:
                return 0.0
            pa, pb = pa[m], pb[m]
        else:
            pa, pb = ta.bbox.center.reshape(1, 2), tb.bbox.center.reshape(1, 2)
        d = float(np.median(epipolar_distance(self.F[(ta.camera_id, tb.camera_id)], pa, pb)))
        geo = max(0.0, 1 - d / self.tau)
        app = 0.0
        ea = ta.detection.embedding if ta.detection is not None else None
        eb = tb.detection.embedding if tb.detection is not None else None
        if ea is not None and eb is not None:
            app = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
            return self.wg * geo + self.wa * max(app, 0.0)
        return geo

    def associate(self, tracks_per_cam: dict[str, list[Track]]) -> list[dict[str, Track]]:
        """Returns clusters: list of {camera_id: Track} — same physical entity."""
        edges = []
        cams = list(tracks_per_cam)
        for a, b in itertools.combinations(cams, 2):
            for ta in tracks_per_cam[a]:
                for tb in tracks_per_cam[b]:
                    if ta.class_name != tb.class_name:
                        continue
                    s = self.affinity(ta, tb)
                    if s >= self.min_aff:
                        edges.append((s, ta, tb))
        edges.sort(key=lambda e: -e[0])
        cluster_of: dict[tuple, int] = {}
        clusters: list[dict[str, Track]] = []
        for s, ta, tb in edges:
            ka, kb = (ta.camera_id, ta.local_track_id), (tb.camera_id, tb.local_track_id)
            ca, cb = cluster_of.get(ka), cluster_of.get(kb)
            if ca is None and cb is None:
                clusters.append({ta.camera_id: ta, tb.camera_id: tb})
                cluster_of[ka] = cluster_of[kb] = len(clusters) - 1
            elif ca is not None and cb is None and tb.camera_id not in clusters[ca]:
                clusters[ca][tb.camera_id] = tb; cluster_of[kb] = ca
            elif cb is not None and ca is None and ta.camera_id not in clusters[cb]:
                clusters[cb][ta.camera_id] = ta; cluster_of[ka] = cb
            elif ca is not None and cb is not None and ca != cb:
                if not set(clusters[ca]) & set(clusters[cb]):
                    clusters[ca].update(clusters[cb])
                    for t in clusters[cb].values():
                        cluster_of[(t.camera_id, t.local_track_id)] = ca
                    clusters[cb] = {}
        # Singletons (seen by one camera only) are kept so identity can persist.
        for c in cams:
            for t in tracks_per_cam[c]:
                if (c, t.local_track_id) not in cluster_of:
                    clusters.append({c: t})
        return [cl for cl in clusters if cl]
