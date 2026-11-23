"""Tiny component registry so configs can select implementations by name."""
from __future__ import annotations

from typing import Callable


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable] = {}

    def register(self, key: str | None = None):
        def deco(obj):
            self._items[key or obj.__name__] = obj
            return obj
        return deco

    def build(self, cfg: dict):
        cfg = dict(cfg)
        key = cfg.pop("type")
        if key not in self._items:
            raise KeyError(f"[{self.name}] unknown type '{key}'. Available: {sorted(self._items)}")
        return self._items[key](**cfg)

    def __contains__(self, key: str) -> bool:
        return key in self._items


DETECTORS = Registry("detector")
POSE_ESTIMATORS = Registry("pose_estimator")
HAND_ESTIMATORS = Registry("hand_estimator")
SEGMENTERS = Registry("segmenter")
TRACKERS = Registry("tracker")
REID_MODELS = Registry("reid")
TRIANGULATORS = Registry("triangulator")
ACTION_MODELS = Registry("action_model")
