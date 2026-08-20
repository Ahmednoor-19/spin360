"""Generate two synthetic 'product' photos (front + back) on a clean white
background, so the pipeline can be demoed without real photography.
Draws a simple bottle with a distinct label on each side.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def bottle(front: bool, size=(512, 512)) -> Image.Image:
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    cx = size[0] // 2
    body = (cx - 90, 150, cx + 90, 470)
    cap = (cx - 40, 90, cx + 40, 150)
    neck = (cx - 30, 130, cx + 30, 170)
    body_color = (54, 122, 196) if front else (196, 84, 54)
    d.rounded_rectangle(body, radius=45, fill=body_color)
    d.rounded_rectangle(neck, radius=8, fill=body_color)
    d.rounded_rectangle(cap, radius=10, fill=(60, 60, 60))
    # label
    label = (cx - 65, 250, cx + 65, 380)
    d.rounded_rectangle(label, radius=12, fill=(245, 245, 245))
    if front:
        d.rectangle((cx - 45, 280, cx + 45, 300), fill=(30, 30, 30))
        d.rectangle((cx - 35, 320, cx + 35, 335), fill=(120, 120, 120))
    else:
        d.ellipse((cx - 30, 290, cx + 30, 350), outline=(30, 30, 30), width=4)
        d.line((cx - 30, 290, cx + 30, 350), fill=(30, 30, 30), width=3)
    return img


def box(front: bool, size=(512, 512)) -> Image.Image:
    """A simple rectangular carton — the easiest shape for 2-view reconstruction
    (near-flat faces, no rounded profile to infer)."""
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    body = (cx - 130, cy - 160, cx + 130, cy + 160)
    body_color = (222, 179, 64) if front else (70, 130, 105)
    d.rectangle(body, fill=body_color)
    d.rectangle((body[0], body[1], body[2], body[1] + 14), fill=(0, 0, 0, 0))
    # panel
    panel = (cx - 95, cy - 90, cx + 95, cy + 70)
    d.rectangle(panel, fill=(250, 250, 248))
    if front:
        d.rectangle((cx - 70, cy - 60, cx + 70, cy - 20), fill=(30, 30, 30))
        for yy in (cy + 5, cy + 25, cy + 45):
            d.line((cx - 60, yy, cx + 60, yy), fill=(120, 120, 120), width=6)
    else:
        d.rectangle((cx - 55, cy - 55, cx + 55, cy + 55), outline=(30, 30, 30), width=5)
        d.line((cx - 55, cy - 55, cx + 55, cy + 55), fill=(30, 30, 30), width=4)
        d.line((cx - 55, cy + 55, cx + 55, cy - 55), fill=(30, 30, 30), width=4)
    return img


def can(front: bool, size=(512, 512)) -> Image.Image:
    """A cylinder with flat top/bottom rims — a good middle-difficulty case
    between the flat box and the curved-shoulder bottle."""
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    cx = size[0] // 2
    body = (cx - 100, 130, cx + 100, 430)
    body_color = (180, 60, 70) if front else (60, 90, 150)
    d.rounded_rectangle(body, radius=14, fill=body_color)
    d.ellipse((cx - 100, 118, cx + 100, 142), fill=(210, 210, 214))     # top rim
    d.ellipse((cx - 100, 418, cx + 100, 442), fill=(170, 170, 176))    # bottom rim
    label = (cx - 80, 220, cx + 80, 340)
    d.rounded_rectangle(label, radius=8, fill=(248, 248, 246))
    if front:
        d.rectangle((cx - 55, 250, cx + 55, 270), fill=(20, 20, 20))
        d.rectangle((cx - 45, 290, cx + 45, 305), fill=(120, 120, 120))
    else:
        d.polygon([(cx, 235), (cx - 40, 320), (cx + 40, 320)], outline=(20, 20, 20), width=4)
    return img


SHAPES = {"bottle": bottle, "box": box, "can": can}


def main(out_dir: str = "samples", shape: str = "bottle") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    draw = SHAPES[shape]
    draw(True).save(out / "front.png")
    draw(False).save(out / "back.png")
    print(f"wrote {out/'front.png'} and {out/'back.png'}")


if __name__ == "__main__":
    shape = sys.argv[2] if len(sys.argv) > 2 else "bottle"
    main(sys.argv[1] if len(sys.argv) > 1 else "samples", shape)
