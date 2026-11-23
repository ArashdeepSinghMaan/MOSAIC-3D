"""Open and read multiple camera streams (files, RTSP, USB) with health monitoring."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class CameraSource:
    camera_id: str
    uri: str | int                 # file path, rtsp url, or device index
    fps: float | None = None
    resolution: tuple[int, int] | None = None   # (w, h)


@dataclass
class CameraHealth:
    frames_read: int = 0
    frames_failed: int = 0
    last_timestamp: float = 0.0
    fps_estimate: float = 0.0
    timestamps: list[float] = field(default_factory=list)


class CameraManager:
    """Owns one cv2.VideoCapture per camera; yields (camera_id, frame_id, ts, frame)."""

    def __init__(self, sources: list[CameraSource]):
        self.sources = {s.camera_id: s for s in sources}
        self.captures: dict[str, cv2.VideoCapture] = {}
        self.health: dict[str, CameraHealth] = {s.camera_id: CameraHealth() for s in sources}
        self.frame_ids: dict[str, int] = {s.camera_id: 0 for s in sources}

    def open(self) -> None:
        for cid, src in self.sources.items():
            cap = cv2.VideoCapture(src.uri)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {cid}: {src.uri}")
            if src.resolution:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, src.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src.resolution[1])
            self.captures[cid] = cap
            log.info("Opened %s (%s)", cid, src.uri)

    def is_file_source(self, camera_id: str) -> bool:
        return isinstance(self.sources[camera_id].uri, str) and not str(
            self.sources[camera_id].uri).startswith(("rtsp://", "http://", "https://"))

    def read(self, camera_id: str) -> tuple[int, float, np.ndarray] | None:
        cap = self.captures[camera_id]
        ok, frame = cap.read()
        h = self.health[camera_id]
        if not ok:
            h.frames_failed += 1
            return None
        fid = self.frame_ids[camera_id]
        self.frame_ids[camera_id] += 1
        # Files: use container PTS; live: use wall clock.
        ts = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if self.is_file_source(camera_id)
              else time.time())
        h.frames_read += 1
        h.timestamps = (h.timestamps + [ts])[-60:]
        if len(h.timestamps) > 1:
            span = h.timestamps[-1] - h.timestamps[0]
            h.fps_estimate = (len(h.timestamps) - 1) / span if span > 0 else 0.0
        h.last_timestamp = ts
        return fid, ts, frame

    def read_all(self) -> dict[str, tuple[int, float, np.ndarray]]:
        out = {}
        for cid in self.captures:
            r = self.read(cid)
            if r is not None:
                out[cid] = r
        return out

    def health_report(self) -> dict[str, dict]:
        return {cid: {"frames_read": h.frames_read, "frames_failed": h.frames_failed,
                      "fps": round(h.fps_estimate, 2), "last_ts": h.last_timestamp}
                for cid, h in self.health.items()}

    def release(self) -> None:
        for cap in self.captures.values():
            cap.release()
        self.captures.clear()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.release()
