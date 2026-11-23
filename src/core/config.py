"""YAML config loading with `_base_` inheritance and dotted overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    for base in cfg.pop("_base_", []) or []:
        cfg = _merge(load_config(path.parent / base), cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg
