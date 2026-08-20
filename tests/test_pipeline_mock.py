"""End-to-end pipeline test (mock provider + cpu render, no GPU/fal).

Runs the real one-shot runner in an isolated subprocess with a temp data dir,
then asserts a valid MP4 was produced and the job reached DONE.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ensure_samples():
    front = ROOT / "samples" / "front.png"
    back = ROOT / "samples" / "back.png"
    if not (front.exists() and back.exists()):
        subprocess.run([sys.executable, "scripts/make_samples.py", "samples"],
                       cwd=ROOT, check=True)
    return front, back


def test_end_to_end(tmp_path):
    front, back = _ensure_samples()
    env = {
        "PATH": __import__("os").environ["PATH"],
        "SPIN360_DATA_DIR": str(tmp_path),
        "SPIN360_DB_URL": f"sqlite:///{tmp_path/'t.db'}",
        "SPIN360_RECON_PROVIDER": "mock",
        "SPIN360_RENDER": "cpu",
        "SPIN360_ISOLATION": "auto",
    }
    r = subprocess.run([sys.executable, "scripts/run_local.py", str(front), str(back)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr

    # the runner prints the JobRecord JSON block
    block = r.stdout.split("=== JobRecord (s.3 contract) ===", 1)[1]
    rec = json.loads(block[block.index("{"): block.rindex("}") + 1])
    assert rec["status"] == "done", rec.get("failure_reason")
    assert rec["video_url"] and rec["mesh_url"]
    assert set(rec["stage_timings_ms"]) >= {
        "isolation", "reconstruction", "normalization", "render", "encode", "quality"}
    mp4s = list(tmp_path.rglob("spin.mp4"))
    assert mp4s and mp4s[0].stat().st_size > 1000
