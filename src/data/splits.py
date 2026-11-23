"""Recording-level, subject-independent splits (never frame-level — avoids leakage)."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def make_recording_splits(recordings: list[dict], ratios=(0.7, 0.15, 0.15), group_by: str = "subject_id",
                          seed: int = 0) -> dict[str, list[str]]:
    """recordings: [{'recording_id', 'subject_id', 'environment', ...}]. Groups never cross splits."""
    groups = defaultdict(list)
    for r in recordings:
        groups[r.get(group_by, r["recording_id"])].append(r["recording_id"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    a, b = int(round(ratios[0] * n)), int(round((ratios[0] + ratios[1]) * n))
    parts = {"train": keys[:a], "val": keys[a:b], "test": keys[b:]}
    return {s: sorted(rid for k in ks for rid in groups[k]) for s, ks in parts.items()}


def write_splits(splits: dict[str, list[str]], out_dir: str | Path) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / f"{name}.json").write_text(json.dumps({"recordings": ids}, indent=1))
