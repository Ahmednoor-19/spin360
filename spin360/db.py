from __future__ import annotations

import datetime as dt
import json
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import (JSON, DateTime, Float, String, create_engine, select)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            sessionmaker)

from .config import settings
from .schemas import JobRecord, JobStatus


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, index=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict] = mapped_column(JSON)  # full JobRecord as JSON
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)


_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
_engine = create_engine(settings.database_url, future=True, connect_args=_connect_args)
_Session = sessionmaker(_engine, expire_on_commit=False, future=True)


def init_db() -> None:
    settings.ensure_dirs()
    Base.metadata.create_all(_engine)


@contextmanager
def session() -> Iterator[Session]:
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# --- record <-> row helpers -------------------------------------------------

def save(record: JobRecord) -> None:
    with session() as s:
        row = s.get(Job, record.job_id)
        data = json.loads(record.model_dump_json())
        if row is None:
            row = Job(job_id=record.job_id)
            s.add(row)
        row.status = record.status.value
        row.idempotency_key = record.idempotency_key
        row.quality_score = record.quality_score
        row.cost_usd = record.cost_usd
        row.payload = data


def get(job_id: str) -> Optional[JobRecord]:
    with session() as s:
        row = s.get(Job, job_id)
        return JobRecord(**row.payload) if row else None


def find_by_idempotency(key: str) -> Optional[JobRecord]:
    """Duplicate-submission guard. Returns the existing terminal/in-flight
    job for an identical (front, back, params) hash, if any."""
    with session() as s:
        row = s.execute(
            select(Job).where(Job.idempotency_key == key)
            .order_by(Job.created_at.desc())
        ).scalars().first()
        return JobRecord(**row.payload) if row else None


def daily_cost_usd(day: Optional[dt.date] = None) -> float:
    day = day or dt.date.today()
    start = dt.datetime.combine(day, dt.time.min)
    end = start + dt.timedelta(days=1)
    with session() as s:
        rows = s.execute(
            select(Job.cost_usd).where(Job.created_at >= start, Job.created_at < end)
        ).scalars().all()
        return float(sum(rows))
