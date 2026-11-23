#!/usr/bin/env python
"""Benchmark geometry modules on synthetic data (no GPU needed)."""
import _bootstrap  # noqa: F401

from src.evaluation.benchmark import benchmark
from src.reconstruction.triangulation import triangulate_skeleton
from tests.synthetic import make_rig, project_skeleton, random_skeleton

cams = make_rig(4)
P = [c.P for c in cams.values()]
X = random_skeleton()
kps, conf = project_skeleton(cams, X, noise_px=1.0)
for m in ("linear", "weighted", "ransac", "ransac_nonlinear"):
    r = benchmark(lambda _: triangulate_skeleton(P, kps, conf, m), list(range(50)), warmup=5, cuda_sync=False)
    print(f"{m:18s} {r['mean_ms']:.2f} ms  ({r['fps']:.0f} skeletons/s)")
