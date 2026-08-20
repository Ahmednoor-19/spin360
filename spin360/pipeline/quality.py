from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class QualityReport:
    quality_score: float
    front_iou: float
    back_iou: float
    smoothness: float


def _silhouette_from_frame(frame: np.ndarray, bg_rgb: tuple[int, int, int], tol: int = 18) -> np.ndarray:
    diff = np.abs(frame.astype(np.int16) - np.array(bg_rgb, np.int16)).max(axis=2)
    return (diff > tol).astype(np.uint8)


def _norm_bbox(mask: np.ndarray, size: int = 128) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.array(Image.fromarray((crop * 255).astype(np.uint8)).resize((size, size))) > 127


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = _norm_bbox(a); b = _norm_bbox(b)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def evaluate(frames_dir: Path, front_mask: Path, back_mask: Path,
             bg_color: str) -> QualityReport:
    bg = bg_color.lstrip("#")
    bg_rgb = (int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16))

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise ValueError("no frames to evaluate")
    imgs = [np.array(Image.open(f).convert("RGB")) for f in frames]

    front_sil = _silhouette_from_frame(imgs[0], bg_rgb)
    back_sil = _silhouette_from_frame(imgs[len(imgs) // 2], bg_rgb)
    fm = np.array(Image.open(front_mask).convert("L")) > 127
    bm = np.array(Image.open(back_mask).convert("L")) > 127

    front_iou = _iou(front_sil, fm)
    back_iou = _iou(back_sil, bm)

    # temporal smoothness: 1 - normalized mean abs frame delta
    deltas = [np.abs(imgs[i].astype(np.int16) - imgs[i - 1].astype(np.int16)).mean()
              for i in range(1, len(imgs))]
    jitter = float(np.mean(deltas)) / 255.0 if deltas else 0.0
    smoothness = float(max(0.0, 1.0 - jitter * 4.0))   # scale so gentle motion ~1

    score = 0.7 * ((front_iou + back_iou) / 2.0) + 0.3 * smoothness
    return QualityReport(quality_score=round(score, 3),
                         front_iou=round(front_iou, 3),
                         back_iou=round(back_iou, 3),
                         smoothness=round(smoothness, 3))
