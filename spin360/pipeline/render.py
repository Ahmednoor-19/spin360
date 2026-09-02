from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from ..config import settings


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# --------------------------------------------------------------------------- #
# Production: Blender                                                          #
# --------------------------------------------------------------------------- #
def _render_blender(glb: Path, frames_dir: Path, n_frames: int, res: int, bg: str) -> None:
    # Absolute paths only: Blender (and libraries loaded earlier in-process, e.g.
    # rembg) can leave the CWD in a different place than expected, which silently
    # breaks a relative --out path (frames get written who-knows-where and the
    # encode stage then fails to find them).
    script = Path(__file__).resolve().parent.parent / "blender" / "turntable.py"
    cmd = [settings.blender_bin, "--background", "--python", str(script), "--",
           "--glb", str(glb.resolve()), "--out", str(frames_dir.resolve()),
           "--frames", str(n_frames), "--res", str(res), "--bg", bg.lstrip("#"),
           "--engine", settings.blender_engine, "--samples", str(settings.blender_samples)]
    subprocess.run(cmd, check=True, timeout=settings.stage_timeout_s, cwd=Path.cwd())


# --------------------------------------------------------------------------- #
# Fallback: numpy software rasterizer                                         #
# --------------------------------------------------------------------------- #
def _concat(scene):
    if not isinstance(scene, trimesh.Scene):
        return scene
    try:
        return scene.to_geometry()          # trimesh >= 4.x
    except Exception:
        return trimesh.util.concatenate(list(scene.geometry.values()))


def _load_colored(glb: Path):
    mesh = _concat(trimesh.load(glb, process=False))
    vis = mesh.visual
    vcol = None
    try:
        if isinstance(vis, trimesh.visual.texture.TextureVisuals):
            vcol = np.asarray(vis.to_color().vertex_colors)[:, :3]
        elif getattr(vis, "vertex_colors", None) is not None:
            vcol = np.asarray(vis.vertex_colors)[:, :3]
    except Exception:
        vcol = None
    if vcol is None or len(vcol) != len(mesh.vertices):
        vcol = np.full((len(mesh.vertices), 3), 184, np.uint8)
    return (mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.int64),
            vcol.astype(np.float32) / 255.0)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def _vertex_normals(V, F):
    """Area-weighted per-vertex normals for smooth (Gouraud) shading."""
    vn = np.zeros(V.shape, np.float32)
    tris = V[F]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # area-weighted
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return vn / norms


# soft two-light rig (key + fill) + ambient — reads like a small studio setup,
# tuned so key+fill+ambient stays <= 1.0 (no blown-out white facets)
_KEY = np.array([0.35, 0.55, 1.0], np.float32); _KEY /= np.linalg.norm(_KEY)
_FILL = np.array([-0.6, 0.15, 0.5], np.float32); _FILL /= np.linalg.norm(_FILL)
_AMBIENT = 0.38


def _shade(VN):
    """Per-vertex brightness from the light rig (vectorised over all vertices)."""
    key = np.maximum(0.0, VN @ _KEY)
    fill = np.maximum(0.0, VN @ _FILL)
    return np.clip(_AMBIENT + 0.46 * key + 0.20 * fill, 0.0, 1.0).astype(np.float32)


def _rasterize(V, F, VC, LV, res, bg_rgb):
    """Orthographic z-buffer rasterizer with smooth (interpolated) colour + shading.

    Colours and per-vertex brightness are barycentric-interpolated across each
    triangle, so facets and hard colour steps disappear. Returns float [0,1].
    """
    img = np.zeros((res, res, 3), np.float32) + np.array(bg_rgb, np.float32) / 255.0
    zbuf = np.full((res, res), -np.inf, np.float32)

    span = 1.2  # map [-0.6,0.6] world -> [0,res]
    sx = (V[:, 0] / span + 0.5) * (res - 1)
    sy = (0.5 - V[:, 1] / span) * (res - 1)
    P = np.stack([sx, sy, V[:, 2]], axis=1)

    for f in F:
        p0, p1, p2 = P[f]
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(area) < 1e-7:            # degenerate
            continue
        # front-facing test in WORLD space (viewer at +Z); the screen Y-flip would
        # otherwise invert the winding, so don't cull on screen-area sign.
        v0, v1, v2 = V[f]
        nz = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
        if nz <= 0.0:                   # back-facing -> occluded
            continue

        minx = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        maxx = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), res - 1)
        miny = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        maxy = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), res - 1)
        if minx > maxx or miny > maxy:
            continue

        xs = np.arange(minx, maxx + 1)[None, :]      # (1, w) — broadcast, no alloc of a full grid
        ys = np.arange(miny, maxy + 1)[:, None]      # (h, 1)
        w0 = ((p1[0] - xs) * (p2[1] - ys) - (p2[0] - xs) * (p1[1] - ys)) / area
        w1 = ((p2[0] - xs) * (p0[1] - ys) - (p0[0] - xs) * (p2[1] - ys)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
        yy, xx = np.nonzero(inside)
        zz = z[yy, xx]
        gy, gx = yy + miny, xx + minx
        vis = zz > zbuf[gy, gx]
        if not vis.any():
            continue
        yv, xv = gy[vis], gx[vis]
        b0, b1, b2 = w0[yy, xx][vis], w1[yy, xx][vis], w2[yy, xx][vis]

        c0, c1, c2 = VC[f]                       # smooth colour across the face
        col = (b0[:, None] * c0 + b1[:, None] * c1 + b2[:, None] * c2)
        l0, l1, l2 = LV[f]                        # smooth brightness across the face
        lam = (b0 * l0 + b1 * l1 + b2 * l2)[:, None]

        zbuf[yv, xv] = zz[vis]
        img[yv, xv] = col * lam
    return np.clip(img, 0.0, 1.0)


def _downsample(img_f, ss):
    if ss <= 1:
        return img_f
    r = img_f.shape[0] // ss
    return img_f[:r * ss, :r * ss].reshape(r, ss, r, ss, 3).mean(axis=(1, 3))


def _render_cpu(glb: Path, frames_dir: Path, n_frames: int, res: int, bg: str,
                elevation_deg: float = 12.0) -> None:
    V0, F, VC = _load_colored(glb)
    VN0 = _vertex_normals(V0, F)
    bg_rgb = _hex_rgb(bg)
    ss = max(1, settings.cpu_ssaa)               # supersampling factor (anti-aliasing)
    hi = res * ss
    tilt = _rot_x(np.radians(elevation_deg))
    V0 = V0 - V0.mean(axis=0)                     # recenter for clean rotation
    for i in range(n_frames):
        R = _rot_y(2 * np.pi * i / n_frames).T @ tilt.T
        V = V0 @ R
        VN = VN0 @ R                             # normals follow the same rotation
        LV = _shade(VN)
        frame = _rasterize(V, F, VC, LV, hi, bg_rgb)
        frame = _downsample(frame, ss)
        Image.fromarray((frame * 255).astype(np.uint8)).save(
            frames_dir / f"frame_{i:04d}.png")


# --------------------------------------------------------------------------- #
def render(normalized_glb: Path, workdir: Path, *, duration_s: float, fps: int,
           resolution: int, bg_color: str) -> Path:
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_frames = max(2, int(round(duration_s * fps)))

    backend = settings.render_backend
    if backend == "blender":
        _render_blender(normalized_glb, frames_dir, n_frames, resolution, bg_color)
    elif backend == "cpu":
        _render_cpu(normalized_glb, frames_dir, n_frames, resolution, bg_color)
    else:
        raise ValueError(f"unknown render backend '{backend}'")
    return frames_dir
