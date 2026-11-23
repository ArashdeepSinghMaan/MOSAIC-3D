#!/usr/bin/env python
"""Export detector / pose models to ONNX or TensorRT engines (Phase 8)."""
import argparse

import _bootstrap  # noqa: F401

from src.deployment.onnx_export import export_ultralytics

p = argparse.ArgumentParser()
p.add_argument("--weights", nargs="+", default=["yolo11m.pt", "yolo11m-pose.pt"])
p.add_argument("--format", choices=["onnx", "engine"], default="onnx")
p.add_argument("--fp32", action="store_true")
a = p.parse_args()
for w in a.weights:
    print(export_ultralytics(w, a.format, half=not a.fp32))
