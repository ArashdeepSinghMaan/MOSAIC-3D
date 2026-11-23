"""Reader for one recording in the unified schema (datasets/<env>/recordings/<id>/...)."""
from __future__ import annotations

import json
from pathlib import Path

from src.calibration.coordinate_transforms import CameraModel


class Recording:
    ANNOTATIONS = ["detections", "poses_2d", "poses_3d", "objects_3d", "interactions", "activities"]

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.recording_id = self.root.name

    @property
    def videos(self) -> dict[str, Path]:
        return {p.stem: p for p in sorted((self.root / "videos").glob("*.mp4"))}

    def cameras(self) -> dict[str, CameraModel]:
        return {p.stem: CameraModel.load(p) for p in sorted((self.root / "calibration").glob("*.yaml"))}

    def timestamps(self) -> list[dict]:
        p = self.root / "synchronization" / "timestamps.json"
        return json.loads(p.read_text()) if p.exists() else []

    def annotation(self, name: str):
        assert name in self.ANNOTATIONS, name
        p = self.root / "annotations" / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def metadata(self) -> dict:
        p = self.root / "metadata.json"
        return json.loads(p.read_text()) if p.exists() else {}
