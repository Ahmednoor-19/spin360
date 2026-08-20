from pathlib import Path

import pytest


def test_status_enum_matches_workbook():
    from spin360.schemas import JobStatus
    values = {s.value for s in JobStatus}
    assert {"queued", "isolating", "reconstructing", "normalizing", "rendering",
            "encoding", "done", "failed"} <= values


def test_job_record_contract_fields():
    from spin360.schemas import JobRecord
    fields = set(JobRecord.model_fields)
    for f in ["job_id", "status", "input_front_url", "input_back_url",
              "object_detected", "detect_confidence", "mesh_url", "video_url",
              "duration_s", "fps", "resolution", "quality_score",
              "failure_reason", "stage_timings_ms"]:
        assert f in fields, f"missing s.3 field: {f}"


def test_idempotency_is_stable_and_param_sensitive(tmp_path):
    from spin360.reliability import idempotency_key
    a = tmp_path / "a.png"; a.write_bytes(b"AAA")
    b = tmp_path / "b.png"; b.write_bytes(b"BBB")
    k1 = idempotency_key(a, b, {"fps": 30})
    k2 = idempotency_key(a, b, {"fps": 30})
    k3 = idempotency_key(a, b, {"fps": 24})
    assert k1 == k2 and k1 != k3


def test_budget_guard():
    from spin360.reliability import check_budget, BudgetError
    from spin360.config import settings
    check_budget(0.10, 0.0)  # ok
    with pytest.raises(BudgetError):
        check_budget(settings.per_job_budget_usd + 1, 0.0)
    with pytest.raises(BudgetError):
        check_budget(0.10, settings.daily_budget_usd)
