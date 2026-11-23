"""Per-module latency profiling for the full multi-camera pipeline."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class PipelineProfiler:
    def __init__(self):
        self.times: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def section(self, name: str):
        t = time.perf_counter()
        yield
        self.times[name].append((time.perf_counter() - t) * 1000)

    def report(self) -> dict[str, dict[str, float]]:
        return {k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95)), "n": len(v)}
                for k, v in self.times.items()}

    def end_to_end_fps(self, key: str = "frame_set") -> float:
        v = self.times.get(key)
        return float(1000 / np.mean(v)) if v else 0.0
