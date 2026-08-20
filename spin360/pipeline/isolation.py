from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import settings


@dataclass
class IsolationResult:
    front_png: Path        # RGBA, background transparent
    back_png: Path
    front_mask: Path       # single-channel mask (255 = object)
    back_mask: Path
    object_detected: bool
    detect_confidence: float


def _confidence_from_mask(mask: np.ndarray) -> float:
    """Heuristic confidence: a clean single blob covering a sane area fraction
    scores high; empty/edge-touching/fragmented masks score low."""
    frac = float((mask > 127).mean())
    if frac < 0.01 or frac > 0.98:
        return 0.05
    # penalise objects that run off all four edges (likely a failed cut)
    edges = mask[[0, -1], :].mean() + mask[:, [0, -1]].mean()
    edge_penalty = min(edges / (255 * 2), 0.4)
    coverage_score = 1.0 - abs(frac - 0.35)      # ~0.35 area is "ideal"
    return float(max(0.0, min(1.0, coverage_score - edge_penalty)))


def _rembg_cut(img_path: Path) -> tuple[Image.Image, np.ndarray]:
    from rembg import remove  # lazy import; heavy
    src = Image.open(img_path).convert("RGBA")
    out = remove(src)                              # returns RGBA with alpha
    alpha = np.array(out.split()[-1])
    return out, alpha


def _grabcut_cut(img_path: Path) -> tuple[Image.Image, np.ndarray]:
    """Dependency-light fallback using OpenCV GrabCut with a centred rect init."""
    import cv2
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise ValueError(f"cannot read image: {img_path}")
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = cv2.GC_FGD
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, binary])
    return Image.fromarray(rgba, "RGBA"), binary


def _cut(img_path: Path) -> tuple[Image.Image, np.ndarray]:
    backend = settings.isolation_backend
    if backend in ("rembg", "auto"):
        try:
            return _rembg_cut(img_path)
        except Exception:
            if backend == "rembg":
                raise
    if backend == "sam2":
        raise NotImplementedError("SAM2 backend not bundled; wire your SAM2 checkpoint here")
    return _grabcut_cut(img_path)


def isolate(front: Path, back: Path, workdir: Path) -> IsolationResult:
    workdir.mkdir(parents=True, exist_ok=True)
    results = {}
    confidences = []
    for name, src in (("front", front), ("back", back)):
        rgba, mask = _cut(Path(src))
        png_path = workdir / f"{name}_cut.png"
        mask_path = workdir / f"{name}_mask.png"
        rgba.save(png_path)
        Image.fromarray(mask).save(mask_path)
        results[name] = (png_path, mask_path)
        confidences.append(_confidence_from_mask(mask))

    conf = float(min(confidences))                 # weakest of the two gates the job
    return IsolationResult(
        front_png=results["front"][0],
        front_mask=results["front"][1],
        back_png=results["back"][0],
        back_mask=results["back"][1],
        object_detected=conf >= settings.detect_confidence_threshold,
        detect_confidence=round(conf, 3),
    )
