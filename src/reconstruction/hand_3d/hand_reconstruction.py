"""Multi-view 3D hand reconstruction (21 landmarks) and hand-to-person assignment.

Hands are associated across views by triangulating the wrist and checking reprojection
consistency; MANO fitting can be added on top of the triangulated landmarks.
"""
from __future__ import annotations


import numpy as np

from src.calibration.coordinate_transforms import CameraModel
from src.core.types import Detection
from src.reconstruction.triangulation.skeleton import triangulate_skeleton
from src.reconstruction.triangulation.weighted import triangulate_weighted


class HandReconstructor:
    def __init__(self, cameras: dict[str, CameraModel], wrist_reproj_thresh: float = 25.0,
                 conf_thresh: float = 0.3):
        self.cameras, self.tau, self.conf_thresh = cameras, wrist_reproj_thresh, conf_thresh

    def _consistent(self, a: tuple[str, Detection], b: tuple[str, Detection]) -> bool:
        (ca, da), (cb, db) = a, b
        Ps = [self.cameras[ca].P, self.cameras[cb].P]
        pts = np.stack([da.keypoints.xy[0], db.keypoints.xy[0]])
        X = triangulate_weighted(Ps, pts, np.ones(2))
        if X is None:
            return False
        errs = [np.linalg.norm(self.cameras[c].project(X)[0] - p) for c, p in zip((ca, cb), pts)]
        return max(errs) < self.tau

    def __call__(self, hands_per_cam: dict[str, list[Detection]]) -> list[dict]:
        """Returns list of {'side', 'joints_3d' (21,3), 'confidence' (21,), 'views'}."""
        results = []
        for side in ("left_hand", "right_hand"):
            items = [(c, h) for c, hs in hands_per_cam.items() for h in hs
                     if h.class_name == side and h.keypoints is not None]
            groups: list[list[tuple[str, Detection]]] = []
            for it in items:
                for g in groups:
                    if it[0] not in {c for c, _ in g} and all(self._consistent(it, o) for o in g):
                        g.append(it); break
                else:
                    groups.append([it])
            for g in groups:
                if len(g) < 2:
                    continue
                P = [self.cameras[c].P for c, _ in g]
                kps = np.stack([h.keypoints.xy for _, h in g])
                conf = np.stack([h.keypoints.confidence for _, h in g])
                X, jc = triangulate_skeleton(P, kps, conf, "weighted", self.conf_thresh)
                results.append({"side": side.split("_")[0], "joints_3d": X, "confidence": jc,
                                "views": [c for c, _ in g]})
        return results


def assign_hands_to_person(hands_3d: list[dict], persons_wrists: dict[int, dict[str, np.ndarray]],
                           max_dist: float = 0.15) -> dict[tuple[int, str], np.ndarray]:
    """Attach 3D hands to persons by distance between hand wrist and body wrist joint."""
    out = {}
    for h in hands_3d:
        wrist = h["joints_3d"][0]
        if np.isnan(wrist).any():
            continue
        best = min(((np.linalg.norm(w[h["side"]] - wrist), pid) for pid, w in persons_wrists.items()
                    if h["side"] in w and not np.isnan(w[h["side"]]).any()), default=(np.inf, None))
        if best[0] <= max_dist and (best[1], h["side"]) not in out:
            out[(best[1], h["side"])] = h["joints_3d"]
    return out
