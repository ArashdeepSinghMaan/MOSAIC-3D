#!/usr/bin/env python
"""Evaluate predicted 3D poses (scenes.jsonl) against unified-schema poses_3d.json."""
import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from src.evaluation.pose_metrics import mpjpe, pa_mpjpe

p = argparse.ArgumentParser()
p.add_argument("--pred", required=True); p.add_argument("--gt", required=True)
p.add_argument("--time-tol", type=float, default=1 / 60)
a = p.parse_args()

gt = json.load(open(a.gt))
pred = [json.loads(line) for line in open(a.pred)]
pairs = []
for g in gt:
    match = min(pred, key=lambda s: abs(s["timestamp"] - g["timestamp"]))
    pp = match["persons"].get(str(g["global_id"]))
    if pp and abs(match["timestamp"] - g["timestamp"]) <= a.time_tol:
        to_arr = lambda j: np.array([[np.nan] * 3 if x is None or None in x else x for x in j], float)  # noqa
        pairs.append((to_arr(pp["pose_3d"]), to_arr(g["joints"])))
P, G = np.stack([x for x, _ in pairs]), np.stack([y for _, y in pairs])
print({"n": len(pairs), "MPJPE_mm": 1000 * mpjpe(P, G), "PA-MPJPE_mm": 1000 * pa_mpjpe(P, G)})
