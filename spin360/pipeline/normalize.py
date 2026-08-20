from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def normalize(raw_glb: Path, workdir: Path, target_size: float = 1.0) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    scene_or_mesh = trimesh.load(raw_glb, process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        try:
            mesh = scene_or_mesh.to_geometry()
        except Exception:
            mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh

    # 1) center on centroid of the bounding box (stable vs. vertex-count changes)
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    mesh.apply_translation(-center)

    # 2) uniform scale so the largest extent == target_size (keeps aspect ratio;
    #    a spin must not distort proportions)
    extent = float((bounds[1] - bounds[0]).max())
    if extent > 0:
        mesh.apply_scale(target_size / extent)

    # 3) up-axis convention: ensure +Y is up. TRELLIS/most GLBs are already
    #    Y-up; this is a guarded no-op that would rotate a Z-up mesh if detected.
    #    (Detection heuristic: if the mesh is much taller in Z than Y, rotate.)
    ext = mesh.bounds[1] - mesh.bounds[0]
    if ext[2] > 1.3 * ext[1]:
        R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(R)

    # 4) rest the object on the ground plane (y = 0) for consistent framing
    mesh.apply_translation([0, -mesh.bounds[0][1], 0])

    out = workdir / "mesh_normalized.glb"
    mesh.export(out)
    return out
