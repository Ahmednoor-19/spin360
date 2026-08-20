from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


@dataclass(frozen=True)
class Settings:
    # --- storage / db -------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("SPIN360_DATA_DIR", "./data")))
    database_url: str = field(default_factory=lambda: _env("SPIN360_DB_URL", "sqlite:///./data/spin360.db"))

    # --- queue ---------------------------------------------------
    redis_url: str = field(default_factory=lambda: _env("SPIN360_REDIS_URL", "redis://localhost:6379/0"))
    # When true, jobs run synchronously in-process (no Redis needed) — handy for
    # the local demo and tests. Production sets this false and runs a worker.
    inline_worker: bool = field(default_factory=lambda: _env("SPIN360_INLINE", "1") == "1")

    # --- reconstruction provider ------------------------------
    # Which implementation of the ReconstructProvider interface to use:
    #   "mock" | "fal_trellis_multi" | "fal_trellis2" | "fal_hunyuan3d"
    reconstruct_provider: str = field(default_factory=lambda: _env("SPIN360_RECON_PROVIDER", "mock"))
    fal_key: str = field(default_factory=lambda: _env("FAL_KEY", ""))
    # Pinned endpoint ids
    fal_trellis_multi_endpoint: str = "fal-ai/trellis/multi"
    fal_trellis2_endpoint: str = "fal-ai/trellis-2"
    fal_hunyuan3d_endpoint: str = "fal-ai/hunyuan3d/v2/multi-view"

    # --- isolation ----------------------------------------------------
    #   "rembg" | "sam2" | "auto" (rembg if available else naive fallback)
    isolation_backend: str = field(default_factory=lambda: _env("SPIN360_ISOLATION", "auto"))

    # --- render -------------------------------------------------------
    #   "blender" (production, needs Blender) | "cpu" (numpy fallback, runs anywhere)
    render_backend: str = field(default_factory=lambda: _env("SPIN360_RENDER", "cpu"))
    blender_bin: str = field(default_factory=lambda: _env("SPIN360_BLENDER_BIN", "blender"))
    #   CYCLES (GPU, headless/Colab-safe) | EEVEE (fast, needs a display)
    blender_engine: str = field(default_factory=lambda: _env("SPIN360_BLENDER_ENGINE", "CYCLES"))
    blender_samples: int = field(default_factory=lambda: int(_env("SPIN360_BLENDER_SAMPLES", "16")))
    #   CPU-render supersampling factor for anti-aliasing (1 = off, 2 = 4x samples/px)
    cpu_ssaa: int = field(default_factory=lambda: int(_env("SPIN360_CPU_SSAA", "2")))

    # --- output defaults ---------------------------------------
    default_duration_s: float = 4.0
    default_fps: int = 30
    default_resolution: int = 512          # square edge in px
    default_bg_color: str = "#FFFFFF"
    min_duration_s: float = 3.0
    max_duration_s: float = 5.0

    # --- quality gate --------------------------------------------
    detect_confidence_threshold: float = _env_float("SPIN360_DETECT_MIN", 0.60)
    quality_score_threshold: float = _env_float("SPIN360_QUALITY_MIN", 0.55)

    # --- economics / guards ------------------------------------------
    per_job_budget_usd: float = _env_float("SPIN360_JOB_BUDGET", 0.30)
    daily_budget_usd: float = _env_float("SPIN360_DAILY_BUDGET", 50.0)
    # rough unit costs used by the budget guard + audit record
    cost_reconstruct_usd: float = 0.12
    cost_render_minute_usd: float = 0.02

    # --- reliability --------------------------------------------------
    max_retries: int = _env_int("SPIN360_MAX_RETRIES", 2)
    retry_base_delay_s: float = 1.0
    stage_timeout_s: int = _env_int("SPIN360_STAGE_TIMEOUT", 1200)
    circuit_fail_threshold: int = 5        # consecutive provider failures -> open

    # --- retention ----------------------------
    artifact_retention_hours: int = _env_int("SPIN360_RETENTION_H", 72)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
