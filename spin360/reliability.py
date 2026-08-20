from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, TypeVar

from .config import settings

T = TypeVar("T")


class BudgetError(RuntimeError):
    """Raised by the budget guard so the orchestrator can fail the job cleanly
    with a user-facing reason."""


class ProviderUnavailable(RuntimeError):
    """Transient provider/outage error — eligible for retry / circuit breaker."""


# --- idempotency -------------------------------

def idempotency_key(front: Path, back: Path, params: dict) -> str:
    """hash(front + back + params). Identical resubmissions map to one job, so
    we never pay twice for the same reconstruction."""
    h = hashlib.sha256()
    for p in (front, back):
        h.update(Path(p).read_bytes())
    for k in sorted(params):
        h.update(f"{k}={params[k]}".encode())
    return h.hexdigest()


# --- retry with exponential backoff -----------------

def with_retry(fn: Callable[[], T], *, retries: int | None = None,
               base_delay: float | None = None,
               retry_on: tuple[type[Exception], ...] = (ProviderUnavailable, TimeoutError),
               on_retry: Callable[[int, Exception], None] | None = None) -> T:
    retries = settings.max_retries if retries is None else retries
    base_delay = settings.retry_base_delay_s if base_delay is None else base_delay
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except retry_on as exc:
            last = exc
            if attempt == retries:
                break
            if on_retry:
                on_retry(attempt + 1, exc)
            time.sleep(base_delay * (2 ** attempt))
    raise last  # type: ignore[misc]


# --- circuit breaker --------------------------------

class CircuitBreaker:
    """Trips open after N consecutive failures; blocks calls for a cooldown so a
    dead provider doesn't take the whole queue with it."""

    def __init__(self, fail_threshold: int | None = None, cooldown_s: float = 30.0):
        self.fail_threshold = fail_threshold or settings.circuit_fail_threshold
        self.cooldown_s = cooldown_s
        self._fails = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.time() - self._opened_at >= self.cooldown_s:
            self._opened_at = None          # half-open: allow one trial
            self._fails = 0
            return False
        return True

    def record_success(self) -> None:
        self._fails = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._fails += 1
        if self._fails >= self.fail_threshold:
            self._opened_at = time.time()


# --- budget guard -------------------------

def check_budget(projected_job_cost: float, daily_spent: float) -> None:
    if projected_job_cost > settings.per_job_budget_usd:
        raise BudgetError(
            f"projected job cost ${projected_job_cost:.2f} exceeds per-job cap "
            f"${settings.per_job_budget_usd:.2f}")
    if daily_spent + projected_job_cost > settings.daily_budget_usd:
        raise BudgetError(
            f"daily budget ${settings.daily_budget_usd:.2f} would be exceeded "
            f"(spent ${daily_spent:.2f})")
