"""Latency / throughput / VRAM benchmarking for any callable module."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def benchmark(fn: Callable, inputs: list, warmup: int = 10, cuda_sync: bool = True) -> dict:
    sync = lambda: None  # noqa: E731
    mem = lambda: 0.0  # noqa: E731
    try:
        import torch
        if torch.cuda.is_available() and cuda_sync:
            sync = torch.cuda.synchronize
            torch.cuda.reset_peak_memory_stats()
            mem = lambda: torch.cuda.max_memory_allocated() / 2 ** 20  # noqa: E731
    except ImportError:
        pass
    for x in inputs[:warmup]:
        fn(x)
    sync()
    lat = []
    for x in inputs:
        t = time.perf_counter(); fn(x); sync(); lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    return {"mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)), "fps": float(1000 / lat.mean()),
            "peak_vram_mb": float(mem())}
