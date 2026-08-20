from __future__ import annotations

from pathlib import Path

from . import db
from .config import settings
from .observability import log, stage_timer
from .pipeline import encode, isolation, normalize, quality, reconstruct, render
from .reliability import (BudgetError, CircuitBreaker, ProviderUnavailable,
                          check_budget, with_retry)
from .schemas import JobRecord, JobStatus
from .storage import key_to_path, store

# Provider-level circuit breaker shared across jobs (s.9 'Provider outage').
_recon_breaker = CircuitBreaker()


def _job_dir(job_id: str) -> Path:
    d = settings.artifacts_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint(rec: JobRecord, status: JobStatus) -> None:
    rec.status = status
    db.save(rec)


def process(job_id: str) -> JobRecord:
    """Run (or resume) a job to a terminal state. Idempotent per stage."""
    rec = db.get(job_id)
    if rec is None:
        raise KeyError(f"job {job_id} not found")
    if rec.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.NEEDS_REVIEW):
        return rec  # already terminal

    wd = _job_dir(job_id)
    front = key_to_path(rec.input_front_url)
    back = key_to_path(rec.input_back_url)
    p = rec.params

    try:
        # --- budget guard up-front (s.11) -----------------------------------
        provider = reconstruct.get_provider()
        projected = provider.unit_cost_usd + settings.cost_render_minute_usd * (p.duration_s / 60.0)
        check_budget(projected, db.daily_cost_usd())

        # --- 1) isolation ---------------------------------------------------
        with stage_timer(rec, "isolation"):
            _checkpoint(rec, JobStatus.ISOLATING)
            iso = isolation.isolate(front, back, wd / "isolation")
            rec.object_detected = iso.object_detected
            rec.detect_confidence = iso.detect_confidence
            db.save(rec)

        # early HITL branch: low isolation confidence -> human review (s.8)
        if iso.detect_confidence < settings.detect_confidence_threshold:
            rec.failure_reason = (f"low isolation confidence "
                                  f"{iso.detect_confidence:.2f} < "
                                  f"{settings.detect_confidence_threshold:.2f}")
            _checkpoint(rec, JobStatus.NEEDS_REVIEW)
            log("job.review", job_id=job_id, reason=rec.failure_reason)
            return rec

        # --- 2) reconstruction (retry + circuit breaker) --------------------
        with stage_timer(rec, "reconstruction"):
            _checkpoint(rec, JobStatus.RECONSTRUCTING)
            if _recon_breaker.is_open:
                raise ProviderUnavailable("reconstruction circuit breaker open")

            def _do_recon():
                return provider.reconstruct(iso.front_png, iso.back_png,
                                            wd / "reconstruct", seed=p.seed)
            try:
                recon = with_retry(
                    _do_recon,
                    on_retry=lambda n, e: log("recon.retry", job_id=job_id, attempt=n, error=str(e)))
                _recon_breaker.record_success()
            except ProviderUnavailable:
                _recon_breaker.record_failure()
                raise
            rec.recon_provider = recon.provider
            rec.recon_model_version = recon.model_version
            rec.seed_used = recon.seed_used
            rec.cost_usd += recon.cost_usd
            rec.mesh_url = store.put_file(f"{job_id}/mesh_raw.glb", recon.glb_path)
            db.save(rec)

        # --- 3) normalization ----------------------------------------------
        with stage_timer(rec, "normalization"):
            _checkpoint(rec, JobStatus.NORMALIZING)
            norm_glb = normalize.normalize(recon.glb_path, wd / "normalize")

        # --- 4) render ------------------------------------------------------
        with stage_timer(rec, "render"):
            _checkpoint(rec, JobStatus.RENDERING)
            frames_dir = render.render(
                norm_glb, wd / "render",
                duration_s=p.duration_s, fps=p.fps,
                resolution=p.resolution, bg_color=p.bg_color)
            rec.cost_usd += settings.cost_render_minute_usd * (p.duration_s / 60.0)

        # --- 5) encode ------------------------------------------------------
        with stage_timer(rec, "encode"):
            _checkpoint(rec, JobStatus.ENCODING)
            mp4 = encode.encode(frames_dir, wd / "encode", fps=p.fps)
            rec.video_url = store.put_file(f"{job_id}/spin.mp4", mp4)
            rec.duration_s = p.duration_s
            rec.fps = p.fps
            rec.resolution = f"{p.resolution}x{p.resolution}"

        # --- 6) quality gate (s.8 auto-action rule) -------------------------
        with stage_timer(rec, "quality"):
            report = quality.evaluate(frames_dir, iso.front_mask, iso.back_mask,
                                      p.bg_color)
            rec.quality_score = report.quality_score
            log("quality", job_id=job_id, **report.__dict__)

        if rec.quality_score >= settings.quality_score_threshold:
            _checkpoint(rec, JobStatus.DONE)
            log("job.done", job_id=job_id, quality=rec.quality_score, cost=rec.cost_usd)
        else:
            rec.failure_reason = (f"quality {rec.quality_score:.2f} < "
                                  f"threshold {settings.quality_score_threshold:.2f}")
            _checkpoint(rec, JobStatus.NEEDS_REVIEW)  # broken spin never auto-shipped
            log("job.review", job_id=job_id, reason=rec.failure_reason)
        return rec

    except BudgetError as e:
        rec.failure_reason = f"budget: {e}"
        _checkpoint(rec, JobStatus.FAILED)
        log("job.failed", job_id=job_id, reason=rec.failure_reason)
        return rec
    except Exception as e:  # any stage failure -> clean terminal with reason
        rec.failure_reason = f"{type(e).__name__}: {e}"
        _checkpoint(rec, JobStatus.FAILED)
        log("job.failed", job_id=job_id, reason=rec.failure_reason)
        return rec
