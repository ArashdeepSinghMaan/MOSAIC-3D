"""Build and run TensorRT engines from ONNX (FP16 / INT8)."""
from __future__ import annotations

from pathlib import Path


def build_engine(onnx_path: str | Path, engine_path: str | Path, fp16: bool = True, int8: bool = False,
                 calibrator=None, workspace_gb: int = 4) -> Path:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calibrator
    engine = builder.build_serialized_network(network, cfg)
    Path(engine_path).write_bytes(engine)
    return Path(engine_path)
