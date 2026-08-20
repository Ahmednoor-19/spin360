"""Async job queue.

Reconstruction + render are long GPU jobs that must not block the HTTP request,
so submission enqueues work and returns immediately. RQ + Redis in production;
an in-process runner when `settings.inline_worker` is set (demo/tests/CI) so the
system runs with zero infra.
"""
from __future__ import annotations

from .config import settings
from .observability import log


def enqueue(job_id: str) -> None:
    if settings.inline_worker:
        # Run in a background thread so POST returns immediately and clients can
        # poll live stage transitions (orchestrator checkpoints status per stage).
        import threading

        from . import orchestrator

        def _run() -> None:
            try:
                orchestrator.process(job_id)
            except Exception as e:  # pragma: no cover - defensive; process logs its own
                log("queue.inline_error", job_id=job_id, error=repr(e))

        log("queue.inline", job_id=job_id)
        threading.Thread(target=_run, name=f"job-{job_id[:8]}", daemon=True).start()
        return
    from redis import Redis
    from rq import Queue
    q = Queue("spin360", connection=Redis.from_url(settings.redis_url))
    q.enqueue("spin360.orchestrator.process", job_id,
              job_timeout=settings.stage_timeout_s * 6)
    log("queue.enqueued", job_id=job_id)
