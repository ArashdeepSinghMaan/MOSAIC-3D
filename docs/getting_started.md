# Getting started

```bash
pip install -r requirements-minimal.txt   # geometry core + tests, CPU only
pytest                                     # synthetic-rig unit tests
python scripts/benchmark_pipeline.py       # triangulation variants timing

pip install -r requirements.txt            # full stack (GPU)
python scripts/calibrate_cameras.py --camera cam_01 --video calib/cam_01.mp4 --out configs/cameras/cam_01.yaml
python scripts/run_mvp.py --config configs/pipeline/mvp.yaml
```
