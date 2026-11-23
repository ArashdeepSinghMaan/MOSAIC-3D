#!/usr/bin/env python
"""Run the Section-12 MVP on a synchronised multi-camera recording."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.acquisition import CameraManager, CameraSource, FrameSynchronizer
from src.calibration.coordinate_transforms import load_camera_rig
from src.core.config import load_config
from src.core.registry import DETECTORS, POSE_ESTIMATORS
from src.perception import PerceptionPipeline
from src.pipeline import MVPPipeline
from src.reconstruction.scene_fusion import SceneBuilder
import src.perception.detection  # noqa: F401  (register models)
import src.perception.human_pose  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/pipeline/mvp.yaml")
p.add_argument("--max-frames", type=int, default=None)
p.add_argument("overrides", nargs="*", help="key.sub=value")
a = p.parse_args()

cfg = load_config(a.config, a.overrides)
rig = load_config(cfg["cameras"])
cams = load_camera_rig(rig["calibration_dir"])
sources = [CameraSource(**s) for s in rig["sources"]]
det = DETECTORS.build(cfg["perception"]["detector"])
pose = POSE_ESTIMATORS.build(cfg["perception"]["pose"])
perc = {s.camera_id: PerceptionPipeline(det, pose) for s in sources}
pipe = MVPPipeline({s.camera_id: cams[s.camera_id] for s in sources}, perc, cfg)
sync = FrameSynchronizer([s.camera_id for s in sources], rig["synchronization"]["tolerance_s"],
                         time_offsets=rig["synchronization"].get("time_offsets"))

out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
n = 0
with CameraManager(sources) as cm, open(out / "scenes.jsonl", "w") as f:
    while a.max_frames is None or n < a.max_frames:
        frames = cm.read_all()
        if not frames:
            break
        for cid, (fid, ts, img) in frames.items():
            sync.push(cid, fid, ts, img)
        while (fs := sync.pop()) is not None:
            scene = pipe.step(fs)
            f.write(json.dumps(SceneBuilder.to_json(scene)) + "\n")
            n += 1
pipe.memory.flush(); pipe.memory.save(str(out / "events.json"))
print(json.dumps(pipe.prof.report(), indent=1))
print("dropped frames:", sync.dropped)
