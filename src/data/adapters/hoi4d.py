"""Adapter: HOI4D. Egocentric RGB-D; MANO hand params and per-object 6D pose."""
from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter


class HOI4DAdapter(DatasetAdapter):
    name = "hoi4d"
    modules = ["hand_pose", "object_pose", "hoi"]
    skeleton = "mano"
    units = "m"

    def list_recordings(self) -> list[str]:
        return sorted(p.name for p in self.source_root.iterdir() if p.is_dir())

    def convert(self, recording_id: str, out_root: str | Path) -> Path:
        out = Path(out_root) / recording_id
        for sub in ("videos", "calibration", "synchronization", "annotations"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        # TODO: implement — map source calibration to CameraModel YAML (X_cam = R X_world + t, metres),
        # convert annotations to datasets/unified_schema/*.schema.json formats.
        raise NotImplementedError(f"{self.name} conversion not implemented yet")
