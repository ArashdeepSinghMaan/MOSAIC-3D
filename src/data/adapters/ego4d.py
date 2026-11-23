"""Adapter: Ego4D. Egocentric long-form video; use FHO / LTA benchmark annotations."""
from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter


class Ego4DAdapter(DatasetAdapter):
    name = "ego4d"
    modules = ["long_term_memory", "anticipation"]
    skeleton = "none"
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
