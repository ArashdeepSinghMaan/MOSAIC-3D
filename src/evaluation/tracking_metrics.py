"""MOT metrics. Uses `motmetrics` / TrackEval when installed; small built-ins otherwise."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def id_switches(gt_ids_per_frame: list[list[int]], pred_ids_per_frame: list[list[int]]) -> int:
    """Count ID switches given per-frame matched (gt_id, pred_id) lists of equal length."""
    last = {}
    sw = 0
    for gts, prs in zip(gt_ids_per_frame, pred_ids_per_frame):
        for g, p in zip(gts, prs):
            if g in last and last[g] != p:
                sw += 1
            last[g] = p
    return sw


def compute_mot_metrics(gt: dict[int, list[tuple[int, np.ndarray]]],
                        pred: dict[int, list[tuple[int, np.ndarray]]], iou_thresh: float = 0.5) -> dict:
    """gt/pred: frame -> [(id, xyxy)]. Returns MOTA, IDF1, IDSW, etc. via motmetrics."""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for f in sorted(set(gt) | set(pred)):
        g, p = gt.get(f, []), pred.get(f, [])
        gb = np.array([b for _, b in g]).reshape(-1, 4); pb = np.array([b for _, b in p]).reshape(-1, 4)
        to_xywh = lambda b: np.c_[b[:, :2], b[:, 2:] - b[:, :2]]  # noqa: E731
        dist = mm.distances.iou_matrix(to_xywh(gb), to_xywh(pb), max_iou=1 - iou_thresh)
        acc.update([i for i, _ in g], [i for i, _ in p], dist)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "motp", "idf1", "num_switches", "num_fragmentations",
                                 "precision", "recall"], name="all")
    return {k: float(v) for k, v in s.loc["all"].items()}


def global_idf1(gt_global: dict, pred_global: dict) -> float:
    """Cross-camera IDF1: treat (camera, frame) pairs as frames of one sequence."""
    tp = defaultdict(int)
    n_gt = sum(len(v) for v in gt_global.values()); n_pr = sum(len(v) for v in pred_global.values())
    for key, gts in gt_global.items():
        for gid, pid in zip(gts, pred_global.get(key, [])):
            tp[(gid, pid)] += 1
    # best one-to-one mapping
    from scipy.optimize import linear_sum_assignment
    g_ids = sorted({g for g, _ in tp}); p_ids = sorted({p for _, p in tp})
    if not g_ids:
        return 0.0
    M = np.zeros((len(g_ids), len(p_ids)))
    for (g, p), c in tp.items():
        M[g_ids.index(g), p_ids.index(p)] = c
    r, c = linear_sum_assignment(-M)
    idtp = M[r, c].sum()
    return float(2 * idtp / (n_gt + n_pr)) if n_gt + n_pr else 0.0
