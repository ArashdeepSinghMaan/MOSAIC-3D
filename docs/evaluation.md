# Evaluation protocol

Every module is evaluated independently (`src/evaluation/`), and failure slices are reported with
`src/evaluation/failure_analysis.py`. Each experiment folder under `experiments/` records config,
dataset version, split, accuracy, latency, VRAM, throughput and failure cases (Section 8).
