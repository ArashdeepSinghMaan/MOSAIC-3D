#!/usr/bin/env python
"""Recording-level, subject-independent train/val/test splits."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.splits import make_recording_splits, write_splits

p = argparse.ArgumentParser()
p.add_argument("--root", required=True, help="folder containing */recordings/*/metadata.json")
p.add_argument("--out", default="datasets/splits"); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
recs = [json.loads(m.read_text()) for m in Path(a.root).glob("**/recordings/*/metadata.json")]
for r in recs:
    r.setdefault("subject_id", "+".join(r.get("subject_ids", [])) or r["recording_id"])
s = make_recording_splits(recs, seed=a.seed)
write_splits(s, a.out)
print({k: len(v) for k, v in s.items()})
