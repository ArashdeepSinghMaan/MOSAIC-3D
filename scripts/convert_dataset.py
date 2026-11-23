#!/usr/bin/env python
"""python scripts/convert_dataset.py --dataset cmu_panoptic --src /data/panoptic --out datasets/general/recordings"""
import argparse

import _bootstrap  # noqa: F401

from src.data.adapters import ADAPTERS

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
p.add_argument("--src", required=True); p.add_argument("--out", required=True)
p.add_argument("--limit", type=int)
a = p.parse_args()
for path in ADAPTERS[a.dataset](a.src).convert_all(a.out, a.limit):
    print("wrote", path)
