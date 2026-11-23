"""FastAPI service: health, scene state, event timeline, offline job submission."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MOSAIC-3D", version="0.1.0")

_STATE: dict = {"latest_scene": None, "timeline": {}, "jobs": {}}


class JobRequest(BaseModel):
    recording_dir: str
    config: str = "configs/pipeline/mvp.yaml"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scene/latest")
def latest_scene():
    if _STATE["latest_scene"] is None:
        raise HTTPException(404, "No scene processed yet")
    return _STATE["latest_scene"]


@app.get("/timeline/{person_id}")
def timeline(person_id: int):
    return _STATE["timeline"].get(person_id, [])


@app.post("/jobs")
def submit_job(req: JobRequest):
    job_id = f"job_{len(_STATE['jobs']) + 1:04d}"
    _STATE["jobs"][job_id] = {"status": "queued", **req.model_dump()}
    # TODO: dispatch to a worker (e.g. BackgroundTasks / Celery) running src.pipeline.MVPPipeline
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _STATE["jobs"]:
        raise HTTPException(404, "Unknown job")
    return _STATE["jobs"][job_id]
