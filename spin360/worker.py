"""Background worker entrypoint.

Run:  python -m spin360.worker
Consumes the 'spin360' queue and executes orchestrator.process for each job.
"""
from __future__ import annotations

from redis import Redis
from rq import Queue, Worker

from .config import settings
from .db import init_db


def main() -> None:
    init_db()
    conn = Redis.from_url(settings.redis_url)
    worker = Worker([Queue("spin360", connection=conn)], connection=conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
