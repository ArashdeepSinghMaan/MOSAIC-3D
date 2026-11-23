"""Record synchronised multi-camera sessions into the unified dataset layout."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from .synchronization import SyncedFrameSet


class MultiCameraRecorder:
    def __init__(self, out_dir: str | Path, fps: float = 30.0, codec: str = "mp4v"):
        self.out_dir = Path(out_dir)
        (self.out_dir / "videos").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "synchronization").mkdir(parents=True, exist_ok=True)
        self.fps, self.fourcc = fps, cv2.VideoWriter_fourcc(*codec)
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.timestamps: list[dict] = []

    def write(self, fs: SyncedFrameSet) -> None:
        for cid, frame in fs.frames.items():
            if cid not in self.writers:
                h, w = frame.shape[:2]
                self.writers[cid] = cv2.VideoWriter(str(self.out_dir / "videos" / f"{cid}.mp4"),
                                                    self.fourcc, self.fps, (w, h))
            self.writers[cid].write(frame)
        self.timestamps.append({"t": fs.timestamp, "frame_ids": fs.frame_ids, "offsets": fs.offsets})

    def close(self) -> None:
        for w in self.writers.values():
            w.release()
        (self.out_dir / "synchronization" / "timestamps.json").write_text(
            json.dumps(self.timestamps, indent=1))
