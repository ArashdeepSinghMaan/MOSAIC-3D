.PHONY: install install-min test lint run-mvp api docker
install:      ; pip install -r requirements.txt && pip install -e .
install-min:  ; pip install -r requirements-minimal.txt && pip install -e .
test:         ; pytest
lint:         ; ruff check src tests scripts
run-mvp:      ; python scripts/run_mvp.py --config configs/pipeline/mvp.yaml
api:          ; uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000
docker:       ; docker build -t mosaic3d:latest .
