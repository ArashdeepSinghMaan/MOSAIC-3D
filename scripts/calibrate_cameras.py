#!/usr/bin/env python
"""Intrinsic calibration per camera from a folder of board images or a video.

python scripts/calibrate_cameras.py --camera cam_01 --video calib/cam_01.mp4 --board charuco \
    --out configs/cameras/cam_01.yaml
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np

from src.calibration.coordinate_transforms import CameraModel
from src.calibration.intrinsic_calibration import (calibrate_charuco, calibrate_checkerboard,
                                                   frames_from_video, load_images)

p = argparse.ArgumentParser()
p.add_argument("--camera", required=True)
p.add_argument("--images"); p.add_argument("--video")
p.add_argument("--board", choices=["checkerboard", "charuco"], default="charuco")
p.add_argument("--out", required=True)
a = p.parse_args()

imgs = frames_from_video(a.video) if a.video else load_images(a.images)
res = calibrate_checkerboard(imgs) if a.board == "checkerboard" else calibrate_charuco(imgs)
print(f"{a.camera}: RMS reprojection error = {res.rms_reprojection_error:.3f} px over {res.n_views} views")
CameraModel(a.camera, res.K, res.dist, np.eye(3), np.zeros(3), res.image_size).save(a.out)
print("saved", a.out)
