"""BoT-SORT via the `boxmot` package (camera-motion compensation + Re-ID). Optional dependency."""
from __future__ import annotations

import numpy as np

from src.core.registry import TRACKERS
from src.core.types import BBox, Detection, Track
from .base import BaseTracker


@TRACKERS.register("botsort")
class BoTSORTTracker(BaseTracker):
    def __init__(self, camera_id: str = "cam_00", reid_weights: str = "osnet_x0_25_msmt17.pt",
                 device: str = "cuda:0", half: bool = True, class_names: list[str] | None = None):
        super().__init__(camera_id)
        from pathlib import Path
        from boxmot import BotSort
        self.tracker = BotSort(reid_weights=Path(reid_weights), device=device, half=half)
        self.class_names = class_names or ["person"]

    def reset(self):
        self.tracker.reset() if hasattr(self.tracker, "reset") else None

    def update(self, detections: list[Detection], image: np.ndarray | None = None) -> list[Track]:
        if image is None:
            raise ValueError("BoT-SORT needs the frame for Re-ID / CMC")
        cls_idx = {c: i for i, c in enumerate(self.class_names)}
        dets = np.array([[*d.bbox.xyxy, d.confidence, cls_idx.get(d.class_name, 0)]
                         for d in detections]).reshape(-1, 6)
        res = self.tracker.update(dets, image)  # (N, 8): x1,y1,x2,y2,id,conf,cls,det_idx
        out = []
        for x1, y1, x2, y2, tid, conf, cls, di in res:
            d = detections[int(di)] if int(di) < len(detections) else None
            if d is not None:
                d.local_track_id = int(tid)
            out.append(Track(self.camera_id, int(tid), self.class_names[int(cls)],
                             BBox(x1, y1, x2, y2), float(conf), detection=d))
        return out
