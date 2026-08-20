"""Headless Blender turntable renderer.

Run:  blender --background --python turntable.py -- \
          --glb mesh.glb --out frames/ --frames 120 --res 512 --bg FFFFFF

Deterministic camera orbit + fixed three-point-ish lighting so every frame is
consistent (s.4 "Deterministic camera orbit + controlled lighting"). Emits
frame_0000.png ... frame_(N-1).png, a full 360° sweep suitable for a seamless
loop (the encoder does not repeat the first frame).

This module is executed *by Blender's own Python* (bpy), not the app venv.
"""
import argparse
import math
import sys

try:
    import bpy
    import mathutils
except ImportError:  # allows importing the file outside Blender without crashing
    bpy = None


def _argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--bg", default="FFFFFF")
    p.add_argument("--engine", default="CYCLES")   # CYCLES is headless
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--camera-elevation", type=float, default=12.0)  # degrees
    return p.parse_args(_argv_after_dashes())


def _setup_engine(scene, engine: str, samples: int) -> None:
    """Select render engine. CYCLES + GPU is the reliable headless/Colab path;
    EEVEE is faster on a workstation with a display but flaky without one.

    Prints the chosen device so the Colab/host log shows GPU vs CPU — a silent
    CPU fallback is the usual cause of a slow (timing-out) render.
    """
    if engine.upper() == "CYCLES":
        scene.render.engine = "CYCLES"
        cyc = scene.cycles
        cyc.samples = samples
        try:
            cyc.use_adaptive_sampling = True          # stop early on clean pixels
            cyc.adaptive_threshold = 0.02
        except Exception:
            pass
        try:                                          # denoise -> few samples look clean
            cyc.use_denoising = True
        except Exception:
            pass

        prefs = bpy.context.preferences.addons["cycles"].preferences
        picked, gpu_names = None, []
        for backend in ("OPTIX", "CUDA", "HIP", "METAL"):
            try:
                prefs.compute_device_type = backend
                try:
                    prefs.refresh_devices()
                except Exception:
                    pass
                prefs.get_devices()
                names = [d.name for d in prefs.devices if d.type == backend]
                if names:
                    picked, gpu_names = backend, names
                    break
            except Exception:
                continue

        if picked:
            for d in prefs.devices:                   # enable GPU devices only
                d.use = (d.type == picked)
            cyc.device = "GPU"
            try:
                cyc.denoiser = "OPTIX" if picked == "OPTIX" else "OPENIMAGEDENOISE"
            except Exception:
                pass
            print(f"[spin360] Cycles GPU: {picked} -> {gpu_names}", flush=True)
        else:
            cyc.device = "CPU"
            print("[spin360] WARNING: no GPU detected — Cycles on CPU (slow). "
                  "Check Runtime>T4 GPU, or set SPIN360_RENDER=cpu.", flush=True)
    else:
        for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scene.render.engine = name
                break
            except TypeError:
                continue
        print(f"[spin360] Engine: {scene.render.engine}", flush=True)


def hex_to_linear(hex_str):
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    # sRGB -> linear
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b), 1.0)


def main():
    args = parse_args()

    # clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.glb)

    # frame the imported object
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        raise SystemExit("no mesh imported")

    # world background colour
    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = hex_to_linear(args.bg)
    bg.inputs[1].default_value = 1.0

    # target = centre of object bbox; radius from its size
    import numpy as np  # noqa
    all_pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
    xs = [p.x for p in all_pts]; ys = [p.y for p in all_pts]; zs = [p.z for p in all_pts]
    cx, cy, cz = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
    size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    radius = size * 2.4

    target = bpy.data.objects.new("target", None)
    target.location = (cx, cy, cz)
    bpy.context.scene.collection.objects.link(target)

    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # fixed key + fill + rim lights
    for name, loc, energy in [("key", (3, -3, 4), 800),
                              ("fill", (-3, -2, 2), 300),
                              ("rim", (0, 4, 3), 400)]:
        ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.size = 5
        lo = bpy.data.objects.new(name, ld); lo.location = (cx + loc[0], cy + loc[1], cz + loc[2])
        bpy.context.scene.collection.objects.link(lo)

    scene = bpy.context.scene
    _setup_engine(scene, args.engine, args.samples)
    scene.render.resolution_x = scene.render.resolution_y = args.res
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"

    elev = math.radians(args.camera_elevation)
    for i in range(args.frames):
        ang = 2 * math.pi * i / args.frames
        cam.location = (cx + radius * math.cos(ang) * math.cos(elev),
                        cy + radius * math.sin(ang) * math.cos(elev),
                        cz + radius * math.sin(elev))
        scene.render.filepath = f"{args.out.rstrip('/')}/frame_{i:04d}.png"
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__" and bpy is not None:
    main()
