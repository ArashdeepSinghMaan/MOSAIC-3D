from __future__ import annotations

from pathlib import Path


def export_torch_to_onnx(model, dummy_input, out_path: str | Path, opset: int = 17,
                         dynamic_batch: bool = True, input_names=("input",), output_names=("output",)):
    import torch
    model.eval()
    dyn = {n: {0: "batch"} for n in (*input_names, *output_names)} if dynamic_batch else None
    torch.onnx.export(model, dummy_input, str(out_path), opset_version=opset, input_names=list(input_names),
                      output_names=list(output_names), dynamic_axes=dyn)
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))
    return Path(out_path)


def export_ultralytics(weights: str, fmt: str = "onnx", half: bool = True, imgsz: int = 640):
    """fmt: 'onnx' | 'engine' (TensorRT)."""
    from ultralytics import YOLO
    return YOLO(weights).export(format=fmt, half=half, imgsz=imgsz, dynamic=fmt == "onnx")
