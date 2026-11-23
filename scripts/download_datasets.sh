#!/usr/bin/env bash
# Most datasets require registration / licence acceptance — this script only prints where to get them.
cat <<'TXT'
Recommended initial set (Section 5.6):
  CMU Panoptic Studio  https://domedb.perception.cs.cmu.edu/   (use panoptic-toolbox getData.sh)
  Assembly101          https://assembly-101.github.io/
  HOI4D                https://hoi4d.github.io/
  MOT17                https://motchallenge.net/data/MOT17/
  YCB-Video (BOP)      https://bop.felk.cvut.cz/datasets/
  Ego4D                https://ego4d-data.org/   (ego4d CLI after licence approval)
Place raw downloads under data/raw/<dataset>/ then run scripts/convert_dataset.py.
TXT
