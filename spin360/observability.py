from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from .schemas import JobRecord


def log(event: str, *, job_id: str | None = None, **fields) -> None:
    rec = {"ts": round(time.time(), 3), "event": event}
    if job_id:
        rec["job_id"] = job_id
    rec.update(fields)
    sys.stdout.write(json.dumps(rec, default=str) + "\n")
    sys.stdout.flush()


@contextmanager
def stage_timer(record: JobRecord, stage: str) -> Iterator[None]:
    """Time a stage and write the elapsed ms into record.stage_timings_ms[stage]."""
    start = time.perf_counter()
    log("stage.start", job_id=record.job_id, stage=stage)
    try:
        yield
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        record.stage_timings_ms[stage] = round(elapsed, 1)
        log("stage.error", job_id=record.job_id, stage=stage,
            elapsed_ms=round(elapsed, 1), error=str(exc), error_type=type(exc).__name__)
        raise
    else:
        elapsed = (time.perf_counter() - start) * 1000.0
        record.stage_timings_ms[stage] = round(elapsed, 1)
        log("stage.done", job_id=record.job_id, stage=stage, elapsed_ms=round(elapsed, 1))
