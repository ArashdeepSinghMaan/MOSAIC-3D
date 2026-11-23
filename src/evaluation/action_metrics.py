from __future__ import annotations

import numpy as np


def accuracy(pred, gt) -> float:
    return float((np.asarray(pred) == np.asarray(gt)).mean())


def macro_f1(pred, gt, num_classes: int | None = None) -> float:
    pred, gt = np.asarray(pred), np.asarray(gt)
    classes = range(num_classes) if num_classes else np.unique(np.r_[pred, gt])
    f1s = []
    for c in classes:
        tp = ((pred == c) & (gt == c)).sum(); fp = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def topk_accuracy(logits: np.ndarray, gt, k: int = 5) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([g in row for g, row in zip(gt, topk)]))


def _tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]) + 1e-9)


def segment_map(preds: list[dict], gts: list[dict], tiou: float = 0.5) -> float:
    """Temporal mAP. preds: [{label,start,end,score}], gts: [{label,start,end}]."""
    aps = []
    for c in {g["label"] for g in gts}:
        g = [x for x in gts if x["label"] == c]; used = [False] * len(g)
        p = sorted([x for x in preds if x["label"] == c], key=lambda x: -x.get("score", 1))
        tp = []
        for x in p:
            ious = [_tiou((x["start"], x["end"]), (y["start"], y["end"])) for y in g]
            j = int(np.argmax(ious)) if ious else -1
            if j >= 0 and ious[j] >= tiou and not used[j]:
                used[j] = True; tp.append(1)
            else:
                tp.append(0)
        if not p:
            aps.append(0.0); continue
        tp = np.array(tp); rec = np.cumsum(tp) / len(g); prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
        aps.append(float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 11)])))
    return float(np.mean(aps)) if aps else float("nan")
