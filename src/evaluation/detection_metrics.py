from __future__ import annotations

import numpy as np


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def average_precision(preds: list[tuple[int, float, np.ndarray]], gts: dict[int, list[np.ndarray]],
                      iou_thresh: float = 0.5) -> float:
    """VOC-style AP (101-pt). preds: [(image_id, score, xyxy)], gts: image_id -> [xyxy]."""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return float("nan")
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = []
    for img, _, box in sorted(preds, key=lambda x: -x[1]):
        g = gts.get(img, [])
        ious = [_iou(box, gb) for gb in g]
        j = int(np.argmax(ious)) if ious else -1
        if j >= 0 and ious[j] >= iou_thresh and not used[img][j]:
            used[img][j] = True; tp.append(1)
        else:
            tp.append(0)
    tp = np.array(tp); fp = 1 - tp
    rec = np.cumsum(tp) / n_gt; prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 101)]))


def map_50_95(preds, gts) -> float:
    return float(np.mean([average_precision(preds, gts, t) for t in np.arange(0.5, 0.96, 0.05)]))
