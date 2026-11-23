from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

ADAPTERS: dict[str, type["DatasetAdapter"]] = {}


class DatasetAdapter(ABC):
    name: str = ""
    modules: list[str] = []          # which MOSAIC-3D modules this dataset evaluates
    units: str = "m"
    skeleton: str = "coco17"
    license_note: str = "Check the dataset's licence before redistribution."

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.name:
            ADAPTERS[cls.name] = cls

    def __init__(self, source_root: str | Path):
        self.source_root = Path(source_root)

    @abstractmethod
    def list_recordings(self) -> list[str]: ...

    @abstractmethod
    def convert(self, recording_id: str, out_root: str | Path) -> Path:
        """Write videos/, calibration/, synchronization/, annotations/ under out_root/recording_id."""

    def convert_all(self, out_root: str | Path, limit: int | None = None) -> list[Path]:
        return [self.convert(r, out_root) for r in self.list_recordings()[:limit]]
