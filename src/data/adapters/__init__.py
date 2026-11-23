"""One adapter per public dataset -> MOSAIC-3D unified schema. Datasets are NOT interchangeable:
each adapter documents its coordinate system, units, skeleton and licence."""
from .base import DatasetAdapter, ADAPTERS
from . import cmu_panoptic, assembly101, hoi4d, mot17, ycb_video, ego4d  # noqa: F401 (registration)
