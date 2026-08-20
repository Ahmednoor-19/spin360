from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..config import settings


def encode(frames_dir: Path, workdir: Path, *, fps: int) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "spin.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",       # required for QuickTime/Safari/most players
        "-movflags", "+faststart",   # web streaming: moov atom at front
        "-preset", "medium",
        "-crf", "20",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=settings.stage_timeout_s)
    return out
