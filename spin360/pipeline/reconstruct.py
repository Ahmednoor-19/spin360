from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh
from PIL import Image

from ..config import settings
from ..reliability import ProviderUnavailable


@dataclass
class ReconResult:
    glb_path: Path
    provider: str
    model_version: str
    seed_used: int | None
    cost_usd: float


class ReconstructProvider(Protocol):
    name: str
    model_version: str
    unit_cost_usd: float

    def reconstruct(self, front_png: Path, back_png: Path, workdir: Path,
                    seed: int | None) -> ReconResult: ...


def _composite_white(img: Image.Image) -> np.ndarray:
    """RGBA cut-out over a white background -> RGB array (H, W, 3)."""
    rgba = np.array(img.convert("RGBA"), dtype=np.float32)
    a = rgba[..., 3:4] / 255.0
    rgb = rgba[..., :3] * a + 255.0 * (1 - a)
    return rgb.astype(np.uint8)


def _lathe_body(front_png: Path, back_png: Path, levels: int = 56,
                n_theta: int = 72, round_frac: float = 0.9) -> trimesh.Trimesh:
    """Build a clean *surface of revolution* from the silhouette: measure the
    object's half-width (radius) at each height, smooth it, and revolve that
    profile 360° into a watertight rounded body. The front photo wraps the front
    180°, the back photo the back 180°. Because it's a true lathe there are no
    seams, welds, or z-fighting — a deliberate stand-in for the real
    reconstruction model that renders a convincing product spin with zero GPU.
    """
    front_img = Image.open(front_png).convert("RGBA")
    back_img = Image.open(back_png).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
    alpha = np.array(front_img.split()[-1])
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0:
        raise ValueError("empty silhouette; isolation likely failed")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pw, ph = max(x1 - x0, 1), max(y1 - y0, 1)
    w = pw / ph                                    # world width (height = 1)
    front_rgb = _composite_white(front_img)
    back_rgb = _composite_white(back_img)
    mask = alpha > 20

    # radius profile r(level) + horizontal centre c(level), sampled down the height
    rows = np.clip((y0 + np.linspace(0, 1, levels) * ph).astype(int), 0, mask.shape[0] - 1)
    half_w = np.zeros(levels, np.float32)          # world half-width per level
    cen_px = np.zeros(levels, np.float32)          # silhouette centre (pixels) per level
    for i, ry in enumerate(rows):
        cols = np.where(mask[ry])[0]
        if len(cols) >= 2:
            half_w[i] = (cols[-1] - cols[0]) / 2.0 / ph
            cen_px[i] = (cols[-1] + cols[0]) / 2.0
        else:
            cen_px[i] = (x0 + x1) / 2.0

    # smooth the radius so the lathe is clean; taper the last couple of levels to 0
    k = 2
    r_s = np.array([half_w[max(0, i - k):i + k + 1].mean() for i in range(levels)], np.float32)
    r_s[0] = r_s[-1] = 0.0
    r_s[1] *= 0.5; r_s[-2] *= 0.5
    r_s = np.maximum(r_s, 1e-4)
    wy = 0.5 - np.linspace(0, 1, levels)           # world y per level (top -> bottom)

    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)  # 0 = +Z (front)
    verts, colors = [], []
    def sample(img, ry, u):                        # u in [0,1] across silhouette width
        px = int(np.clip(x0 + u * pw, 0, img.shape[1] - 1))
        return img[int(ry), px]

    for i in range(levels):
        ry = rows[i]
        for th in thetas:
            r = r_s[i]
            x = r * np.sin(th)                     # +sin -> right; front view width correct
            z = r * np.cos(th)                     # +Z toward camera
            verts.append([x, wy[i], z])
            # front photo faces the camera at 0° (the -Z-facing render side); back
            # photo on the opposite hemisphere. u follows sin(th).
            u = 0.5 + 0.5 * np.sin(th)
            colors.append(sample(back_rgb if z >= 0 else front_rgb, ry, u))

    def vid(i, k):
        return i * n_theta + (k % n_theta)

    faces = []
    for i in range(levels - 1):
        for kk in range(n_theta):
            a, b = vid(i, kk), vid(i, kk + 1)
            c, d = vid(i + 1, kk + 1), vid(i + 1, kk)
            faces.append([a, b, c]); faces.append([a, c, d])

    # close the very top and bottom with a small fan to a pole vertex
    top_c = len(verts); verts.append([0.0, wy[0], 0.0]); colors.append(front_rgb[rows[0], (x0 + x1)//2])
    bot_c = len(verts); verts.append([0.0, wy[-1], 0.0]); colors.append(front_rgb[rows[-1], (x0 + x1)//2])
    for kk in range(n_theta):
        faces.append([top_c, vid(0, kk + 1), vid(0, kk)])
        faces.append([bot_c, vid(levels - 1, kk), vid(levels - 1, kk + 1)])

    return trimesh.Trimesh(vertices=np.array(verts, np.float32),
                           faces=np.array(faces, np.int64),
                           vertex_colors=np.array(colors).astype(np.uint8),
                           process=False)


# --------------------------------------------------------------------------- #
# Local mock — real geometry, no external calls                               #
# --------------------------------------------------------------------------- #
class MockProvider:
    name = "mock"
    model_version = "mock-lathe-v2"
    unit_cost_usd = 0.0

    def reconstruct(self, front_png: Path, back_png: Path, workdir: Path,
                    seed: int | None) -> ReconResult:
        workdir.mkdir(parents=True, exist_ok=True)
        mesh = _lathe_body(front_png, back_png)
        out = workdir / "mesh_raw.glb"
        mesh.export(out)
        return ReconResult(glb_path=out, provider=self.name,
                           model_version=self.model_version, seed_used=seed,
                           cost_usd=self.unit_cost_usd)


# --------------------------------------------------------------------------- #
# fal.ai TRELLIS multi-view — production default                              #
# --------------------------------------------------------------------------- #
class FalTrellisMultiProvider:
    name = "fal_trellis_multi"
    unit_cost_usd = settings.cost_reconstruct_usd

    def __init__(self):
        self.endpoint = settings.fal_trellis_multi_endpoint
        self.model_version = self.endpoint  # pinned id doubles as the version tag

    def reconstruct(self, front_png: Path, back_png: Path, workdir: Path,
                    seed: int | None) -> ReconResult:
        workdir.mkdir(parents=True, exist_ok=True)
        if not settings.fal_key:
            raise ProviderUnavailable("FAL_KEY not set")
        try:
            import fal_client
            import requests
        except ImportError as e:  # pragma: no cover
            raise ProviderUnavailable(f"fal deps missing: {e}")

        # 1) upload both cut-outs, 2) submit front+back as image_urls, 3) poll,
        #    4) download the returned GLB. Endpoint shape per fal docs.
        try:
            front_url = fal_client.upload_file(str(front_png))
            back_url = fal_client.upload_file(str(back_png))
            args = {"image_urls": [front_url, back_url]}
            if seed is not None:
                args["seed"] = seed
            result = fal_client.subscribe(self.endpoint, arguments=args, with_logs=False)
        except Exception as e:  # network / provider errors are transient -> retry/breaker
            raise ProviderUnavailable(f"fal call failed: {e}") from e

        mesh_url = (result.get("model_mesh") or {}).get("url")
        if not mesh_url:
            raise ProviderUnavailable(f"fal returned no mesh: {result!r}")
        out = workdir / "mesh_raw.glb"
        resp = requests.get(mesh_url, timeout=settings.stage_timeout_s)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        return ReconResult(glb_path=out, provider=self.name,
                           model_version=self.model_version,
                           seed_used=result.get("seed", seed),
                           cost_usd=self.unit_cost_usd)


_TRELLIS_PIPE = None  # heavy model — load once, reuse across jobs


class LocalTrellisProvider:
    """Self-hosted TRELLIS multi-image (front+back) on a local GPU. """
    name = "local_trellis"
    model_version = "microsoft/TRELLIS-image-large"
    unit_cost_usd = 0.0  # your own GPU time; not metered per-job here

    def _pipeline(self):
        global _TRELLIS_PIPE
        if _TRELLIS_PIPE is None:
            try:
                import torch
                from trellis.pipelines import TrellisImageTo3DPipeline
            except Exception as e:  # pragma: no cover - only on a GPU host
                raise ProviderUnavailable(f"TRELLIS not installed: {e}") from e
            if not torch.cuda.is_available():
                raise ProviderUnavailable("no CUDA device for local TRELLIS")
            pipe = TrellisImageTo3DPipeline.from_pretrained(self.model_version)
            pipe.cuda()
            _TRELLIS_PIPE = pipe
        return _TRELLIS_PIPE

    def reconstruct(self, front_png: Path, back_png: Path, workdir: Path,
                    seed: int | None) -> ReconResult:
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            from trellis.utils import postprocessing_utils
        except Exception as e:  # pragma: no cover
            raise ProviderUnavailable(f"TRELLIS deps missing: {e}") from e

        pipe = self._pipeline()
        front = Image.open(front_png).convert("RGBA")
        back = Image.open(back_png).convert("RGBA")
        try:
            outputs = pipe.run_multi_image(
                [front, back],
                seed=seed if seed is not None else 1,
                mode="stochastic",  # matches the reference multi-image algo
            )
        except Exception as e:  # OOM / kernel issues are retryable via the breaker
            raise ProviderUnavailable(f"TRELLIS inference failed: {e}") from e

        # gaussian carries appearance, mesh carries geometry -> fused textured GLB
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0], outputs["mesh"][0],
            simplify=0.95, texture_size=1024,
        )
        out = workdir / "mesh_raw.glb"
        glb.export(str(out))
        return ReconResult(glb_path=out, provider=self.name,
                           model_version=self.model_version, seed_used=seed,
                           cost_usd=self.unit_cost_usd)


_PROVIDERS = {
    "mock": MockProvider,
    "fal_trellis_multi": FalTrellisMultiProvider,
    "local_trellis": LocalTrellisProvider,
    # "fal_trellis2" / "fal_hunyuan3d" register here once wired — same interface.
}


def get_provider(name: str | None = None) -> ReconstructProvider:
    name = name or settings.reconstruct_provider
    if name not in _PROVIDERS:
        raise ValueError(f"unknown reconstruct provider '{name}'; "
                         f"known: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()
