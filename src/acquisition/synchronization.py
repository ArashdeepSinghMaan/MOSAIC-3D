"""Timestamp-based multi-camera frame synchronisation and dropped-frame detection."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyncedFrameSet:
    timestamp: float                          # reference timestamp
    frames: dict[str, np.ndarray]             # camera_id -> image
    frame_ids: dict[str, int]
    offsets: dict[str, float] = field(default_factory=dict)  # per-cam |dt| to reference

    @property
    def max_offset(self) -> float:
        return max(self.offsets.values(), default=0.0)


class FrameSynchronizer:
    """Buffers frames per camera and emits sets whose timestamps agree within `tolerance`.

    `time_offsets` holds per-camera clock corrections (seconds) estimated offline,
    e.g. from an audio clap, LED flash or hardware trigger log.
    """

    def __init__(self, camera_ids: list[str], tolerance: float = 1 / 60,
                 buffer_size: int = 64, time_offsets: dict[str, float] | None = None,
                 require_all: bool = True):
        self.camera_ids = list(camera_ids)
        self.tolerance = tolerance
        self.require_all = require_all
        self.offsets = time_offsets or {c: 0.0 for c in camera_ids}
        self.buffers: dict[str, deque] = {c: deque(maxlen=buffer_size) for c in camera_ids}
        self.dropped: dict[str, int] = {c: 0 for c in camera_ids}

    def push(self, camera_id: str, frame_id: int, timestamp: float, frame: np.ndarray) -> None:
        self.buffers[camera_id].append((timestamp + self.offsets.get(camera_id, 0.0), frame_id, frame))

    def pop(self) -> SyncedFrameSet | None:
        if any(len(self.buffers[c]) == 0 for c in self.camera_ids):
            return None
        # Reference = latest of the earliest heads -> no camera can still produce an older match.
        ref = max(self.buffers[c][0][0] for c in self.camera_ids)
        frames, fids, offs = {}, {}, {}
        for c in self.camera_ids:
            buf = self.buffers[c]
            # Discard frames that are too old to ever match.
            while len(buf) > 1 and buf[0][0] < ref - self.tolerance:
                buf.popleft()
                self.dropped[c] += 1
            ts, fid, fr = buf[0]
            if abs(ts - ref) <= self.tolerance:
                frames[c], fids[c], offs[c] = fr, fid, abs(ts - ref)
        if self.require_all and len(frames) < len(self.camera_ids):
            return None
        for c in frames:
            self.buffers[c].popleft()
        return SyncedFrameSet(ref, frames, fids, offs)


def detect_dropped_frames(timestamps: np.ndarray, nominal_fps: float, factor: float = 1.5) -> np.ndarray:
    """Indices i where the gap timestamps[i]-timestamps[i-1] exceeds factor * nominal period."""
    gaps = np.diff(np.asarray(timestamps, dtype=np.float64))
    return np.where(gaps > factor / nominal_fps)[0] + 1


def estimate_offset_from_signals(sig_a: np.ndarray, sig_b: np.ndarray, rate: float) -> float:
    """Clock offset (s) of b relative to a via cross-correlation (e.g. mean-intensity flash signal)."""
    a = (sig_a - sig_a.mean()) / (sig_a.std() + 1e-9)
    b = (sig_b - sig_b.mean()) / (sig_b.std() + 1e-9)
    corr = np.correlate(a, b, mode="full")
    lag = np.argmax(corr) - (len(b) - 1)
    return lag / rate
