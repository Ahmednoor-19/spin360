from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    ISOLATING = "isolating"
    RECONSTRUCTING = "reconstructing"
    NORMALIZING = "normalizing"
    RENDERING = "rendering"
    ENCODING = "encoding"
    NEEDS_REVIEW = "needs_review"
    DONE = "done"
    FAILED = "failed"


# Ordered pipeline stages. The status enum above mixes stages + terminal
# states; this list is just the stage progression used for checkpoints/timings.
STAGE_ORDER = [
    JobStatus.ISOLATING,
    JobStatus.RECONSTRUCTING,
    JobStatus.NORMALIZING,
    JobStatus.RENDERING,
    JobStatus.ENCODING,
]


class JobParams(BaseModel):
    """Optional generation params from 'Required inputs'."""
    duration_s: float = 4.0
    fps: int = 30
    resolution: int = 512
    bg_color: str = "#FFFFFF"
    seed: Optional[int] = None


class JobRecord(BaseModel):
    """The 'Structured output schema', field-for-field."""
    job_id: str
    status: JobStatus = JobStatus.QUEUED

    input_front_url: Optional[str] = None
    input_back_url: Optional[str] = None
    params: JobParams = Field(default_factory=JobParams)

    object_detected: Optional[bool] = None
    detect_confidence: Optional[float] = None

    mesh_url: Optional[str] = None
    video_url: Optional[str] = None

    duration_s: Optional[float] = None
    fps: Optional[int] = None
    resolution: Optional[str] = None     

    quality_score: Optional[float] = None
    failure_reason: Optional[str] = None

    # object: per-stage latency for tracing
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)

    # --- audit / reproducibility -----------------------------
    idempotency_key: Optional[str] = None
    recon_provider: Optional[str] = None
    recon_model_version: Optional[str] = None
    seed_used: Optional[int] = None
    cost_usd: float = 0.0

    model_config = {"use_enum_values": False}


# Convenience alias so callers can type-hint the public API response clearly.
JobResponse = JobRecord
