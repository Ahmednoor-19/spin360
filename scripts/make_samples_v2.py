"""Generate two polished synthetic product photo pairs (front+back) for testing
Spin360 — a skincare bottle and a snack box. Plain studio background (isolation-
friendly), soft shadow, gradient body, printed label — closer to a real product
photo than the original placeholder bottle."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples_v2"
OUT.mkdir(exist_ok=True)
W, H = 900, 1200

def studio_bg():
    """Soft vertical studio-grey gradient background (not pure white — helps
    isolation contrast against light products, and looks like a real product shot)."""
    top, bot = np.array([246, 246, 248]), np.array([225, 226, 230])
    grad = np.linspace(0, 1, H)[:, None, None]
    arr = (top * (1 - grad) + bot * grad).astype(np.uint8)
    arr = np.repeat(arr, W, axis=1)
    return Image.fromarray(arr, "RGB")

def add_shadow(base, mask_img, cx, by):
    """Soft elliptical contact shadow under the product."""
    sh = Image.new("L", base.size, 0)
    d = ImageDraw.Draw(sh)
    d.ellipse([cx - 160, by - 28, cx + 160, by + 40], fill=70)
    sh = sh.filter(ImageFilter.GaussianBlur(22))
    base.paste(Image.new("RGB", base.size, (60, 60, 65)), mask=sh)

def font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()

def rounded_gradient_body(draw_img, box, top_color, bot_color, radius=60):
    x0, y0, x1, y1 = box
    grad = np.linspace(0, 1, y1 - y0)[:, None, None]
    top_c, bot_c = np.array(top_color), np.array(bot_color)
    band = (top_c * (1 - grad) + bot_c * grad).astype(np.uint8)
    band = np.repeat(band, x1 - x0, axis=1)
    body = Image.fromarray(band, "RGB")
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, x1 - x0 - 1, y1 - y0 - 1],
                                           radius=radius, fill=255)
    draw_img.paste(body, (x0, y0), mask)
    return mask

# ---------------------------------------------------------------- bottle ----
def make_bottle(front: bool):
    img = studio_bg(); d = ImageDraw.Draw(img)
    cx = W // 2
    body_box = [cx - 190, 430, cx + 190, 1080]
    add_shadow(img, None, cx, 1078)
    cap_box = [cx - 95, 300, cx + 95, 440]
    rounded_gradient_body(img, cap_box, (60, 62, 66), (35, 36, 40), radius=26)
    shoulder = [cx - 190, 400, cx + 190, 470]
    rounded_gradient_body(img, shoulder, (235, 244, 238) if front else (250, 238, 226),
                          (200, 214, 204) if front else (222, 196, 168), radius=90)
    top_c = (232, 246, 238) if front else (248, 226, 198)
    bot_c = (176, 202, 188) if front else (206, 158, 108)
    rounded_gradient_body(img, body_box, top_c, bot_c, radius=55)
    # specular highlight streak
    hl = Image.new("L", img.size, 0)
    ImageDraw.Draw(hl).rounded_rectangle([cx - 150, 470, cx - 110, 1040], radius=20, fill=90)
    hl = hl.filter(ImageFilter.GaussianBlur(18))
    img.paste(Image.new("RGB", img.size, (255, 255, 255)), mask=hl)
    # label panel
    lab = [cx - 140, 620, cx + 140, 900]
    ImageDraw.Draw(img).rounded_rectangle(lab, radius=18, fill=(250, 250, 248),
                                          outline=(210, 210, 205), width=3)
    fnt_big, fnt_sm = font(46), font(28)
    if front:
        d.text((cx, 690), "BOTANIQ", font=fnt_big, fill=(40, 90, 60), anchor="mm")
        d.line([cx - 90, 730, cx + 90, 730], fill=(180, 190, 180), width=2)
        d.text((cx, 770), "Hydrating Serum", font=fnt_sm, fill=(70, 70, 70), anchor="mm")
        d.text((cx, 850), "50 ml / 1.7 fl oz", font=font(22), fill=(120, 120, 120), anchor="mm")
    else:
        d.text((cx, 660), "DIRECTIONS", font=font(30), fill=(60, 60, 60), anchor="mm")
        lines = ["Apply 2-3 drops to", "clean skin morning", "and night. Avoid", "direct sunlight."]
        for i, ln in enumerate(lines):
            d.text((cx, 710 + i * 34), ln, font=font(22), fill=(90, 90, 90), anchor="mm")
        d.rectangle([cx - 60, 850, cx + 60, 880], fill=(20, 20, 20))
        for i in range(14):
            x = cx - 58 + i * 8
            if i % 2 == 0:
                d.line([x, 852, x, 878], fill=(255, 255, 255), width=3)
    return img

# ------------------------------------------------------------------- box ----
def make_box(front: bool):
    img = studio_bg(); d = ImageDraw.Draw(img)
    cx = W // 2
    box = [cx - 220, 380, cx + 220, 1040]
    add_shadow(img, None, cx, 1038)
    top_c = (222, 90, 70) if front else (70, 120, 190)
    bot_c = (176, 58, 46) if front else (46, 84, 150)
    rounded_gradient_body(img, box, top_c, bot_c, radius=22)
    # edge crease lines (fold lines read as real cardboard geometry)
    for fx in (cx - 220 + 26, cx + 220 - 26):
        d.line([fx, 380, fx, 1040], fill=(0, 0, 0, 40), width=2)
    fnt_big, fnt_sm = font(52), font(26)
    if front:
        d.rectangle([cx - 170, 470, cx + 170, 620], fill=(255, 255, 255))
        d.text((cx, 545), "CRUNCH\nBITES", font=fnt_big, fill=(200, 40, 30),
               anchor="mm", align="center", spacing=6)
        d.ellipse([cx - 90, 680, cx + 90, 860], fill=(230, 180, 60))
        d.text((cx, 770), "OATS", font=font(34), fill=(120, 70, 10), anchor="mm")
        d.text((cx, 970), "NET WT 12 OZ (340g)", font=fnt_sm, fill=(255, 255, 255), anchor="mm")
    else:
        d.rectangle([cx - 190, 420, cx + 190, 1000], fill=(255, 255, 255))
        d.text((cx, 450), "NUTRITION FACTS", font=font(30), fill=(20, 20, 20), anchor="mm")
        d.line([cx - 170, 480, cx + 170, 480], fill=(20, 20, 20), width=3)
        rows = [("Calories", "210"), ("Total Fat", "6g"), ("Sodium", "140mg"),
                ("Total Carb", "34g"), ("Protein", "5g")]
        for i, (k, v) in enumerate(rows):
            y = 520 + i * 46
            d.text((cx - 160, y), k, font=font(24), fill=(30, 30, 30), anchor="lm")
            d.text((cx + 160, y), v, font=font(24), fill=(30, 30, 30), anchor="rm")
        d.line([cx - 170, 800, cx + 170, 800], fill=(20, 20, 20), width=2)
        d.text((cx, 850), "INGREDIENTS: OATS, HONEY,\nALMONDS, SEA SALT",
               font=font(20), fill=(60, 60, 60), anchor="mm", align="center", spacing=6)
    return img

for name, fn in (("bottle", make_bottle), ("box", make_box)):
    fn(True).convert("RGB").save(OUT / f"{name}_front.png")
    fn(False).convert("RGB").save(OUT / f"{name}_back.png")
    print("wrote", name)
