"""HTTP API.

Endpoints:
  POST /jobs                 upload front+back (+ optional params) -> job_id
  GET  /jobs/{job_id}        poll the full JobRecord
  GET  /jobs/{job_id}/video  stream the finished MP4
  GET  /healthz              liveness

"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import db
from .config import settings
from .observability import log
from .queue import enqueue
from .reliability import idempotency_key
from .schemas import JobParams, JobRecord, JobStatus
from .storage import key_to_path, store

app = FastAPI(title="Spin360", version="0.1.0")

_WEB_DIR = Path(__file__).resolve().parent / "web"


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page pitch UI (upload two shots, watch the turntable)."""
    return FileResponse(_WEB_DIR / "index.html", media_type="text/html")


@app.on_event("startup")
def _startup() -> None:
    settings.ensure_dirs()
    db.init_db()
    log("api.startup", provider=settings.reconstruct_provider,
        render=settings.render_backend, inline=settings.inline_worker)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "provider": settings.reconstruct_provider}


def _validate_image(f: UploadFile) -> None:
    if f.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, f"unsupported type {f.content_type}; use PNG/JPG")


@app.post("/jobs")
async def create_job(
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...),
    duration_s: float = Form(settings.default_duration_s),
    fps: int = Form(settings.default_fps),
    resolution: int = Form(settings.default_resolution),
    bg_color: str = Form(settings.default_bg_color),
    seed: int | None = Form(None),
) -> JSONResponse:
    _validate_image(front_image)
    _validate_image(back_image)
    duration_s = min(max(duration_s, settings.min_duration_s), settings.max_duration_s)

    job_id = uuid.uuid4().hex
    front_key = f"{job_id}/input_front.png"
    back_key = f"{job_id}/input_back.png"
    store.put_bytes(front_key, await front_image.read())
    store.put_bytes(back_key, await back_image.read())

    params = JobParams(duration_s=duration_s, fps=fps, resolution=resolution,
                       bg_color=bg_color, seed=seed)
    idem = idempotency_key(store.path(front_key), store.path(back_key),
                           params.model_dump())

    # duplicate-submission guard (s.9): identical inputs+params -> return existing
    existing = db.find_by_idempotency(idem)
    if existing is not None:
        log("job.dedup", job_id=existing.job_id, idempotency_key=idem)
        return JSONResponse(existing.model_dump(mode="json"), status_code=200)

    rec = JobRecord(
        job_id=job_id, status=JobStatus.QUEUED,
        input_front_url=store.url_for(front_key),
        input_back_url=store.url_for(back_key),
        params=params, idempotency_key=idem,
    )
    db.save(rec)
    log("job.created", job_id=job_id, idempotency_key=idem)
    enqueue(job_id)  # inline in demo mode; async via RQ in prod

    return JSONResponse(db.get(job_id).model_dump(mode="json"), status_code=202)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JobRecord:
    rec = db.get(job_id)
    if rec is None:
        raise HTTPException(404, "job not found")
    return rec


@app.get("/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    rec = db.get(job_id)
    if rec is None or not rec.video_url:
        raise HTTPException(404, "video not ready")
    path: Path = key_to_path(rec.video_url)
    if not path.exists():
        raise HTTPException(410, "artifact expired")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
