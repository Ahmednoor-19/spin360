"""Run one Spin360 job end-to-end, in-process, no Redis / no server.

Usage:
    python scripts/run_local.py samples/front.png samples/back.png

Prints the final JobRecord (the s.3 structured contract) and the local MP4 path.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

# allow running as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spin360 import db                      
from spin360.config import settings          
from spin360.orchestrator import process     
from spin360.reliability import idempotency_key  
from spin360.schemas import JobParams, JobRecord, JobStatus  
from spin360.storage import key_to_path, store  


def main(front: str, back: str) -> int:
    settings.ensure_dirs()
    db.init_db()

    job_id = uuid.uuid4().hex
    fk, bk = f"{job_id}/input_front.png", f"{job_id}/input_back.png"
    store.put_bytes(fk, Path(front).read_bytes())
    store.put_bytes(bk, Path(back).read_bytes())

    params = JobParams()
    rec = JobRecord(
        job_id=job_id, status=JobStatus.QUEUED,
        input_front_url=store.url_for(fk), input_back_url=store.url_for(bk),
        params=params,
        idempotency_key=idempotency_key(store.path(fk), store.path(bk),
                                        params.model_dump()),
    )
    db.save(rec)

    final = process(job_id)
    print("\n=== JobRecord (s.3 contract) ===")
    print(json.dumps(final.model_dump(mode="json"), indent=2))

    if final.video_url:
        print("\nMP4 ->", key_to_path(final.video_url))
    return 0 if final.status == JobStatus.DONE else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python scripts/run_local.py FRONT.png BACK.png")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
