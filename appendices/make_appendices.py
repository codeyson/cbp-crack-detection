"""Appendix figures: code and terminal output rendered to PNG with red label boxes.

python appendices/make_appendices.py                          # rebuild app_05, app_06, app_08, app_09
python appendices/make_appendices.py box IN OUT x0 y0 x1 y1 "label" [...]   # red box on your own screenshot
"""
import json, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
RED = (220, 0, 0)
MONO = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
SANS = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 17)
LH, PAD, GUT = 26, 16, 64   # line height, padding, line-number gutter (px)


def render(title, rows, boxes, out, dark=False):
    """rows: [(lineno or None, text)], None lineno = no number, text None = '...' gap.
    boxes: [(first_lineno, last_lineno, label)] in file line numbers."""
    bg, fg, num = ((12, 12, 12), (220, 220, 220), (110, 110, 110)) if dark else ((255, 255, 255), (20, 20, 20), (150, 150, 150))
    texts = [t if t is not None else "   ..." for _, t in rows]
    x1 = GUT + PAD * 2 + int(max(MONO.getlength(t) for t in texts)) + 16    # box right edge, tags go past it
    w = x1 + max([int(SANS.getlength(l)) + 40 for *_, l in boxes] or [PAD])
    top = 44 + PAD
    img = Image.new("RGB", (max(w, 700), top + LH * len(rows) + PAD * 2), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 44], fill=(45, 45, 48))
    d.text((PAD, 12), title, font=SANS, fill=(240, 240, 240))
    y_of = {}
    for i, ((n, _), t) in enumerate(zip(rows, texts)):
        y = top + PAD + i * LH
        if n is not None:
            y_of[n] = y
        if isinstance(n, int):   # str keys: boxable rows without a line number (program output)
            d.text((PAD, y), f"{n:>4}", font=MONO, fill=num)
        d.text((PAD + GUT, y), t.rstrip(), font=MONO, fill=fg if rows[i][1] is not None else num)
    for a, b, label in boxes:
        tag(d, PAD + GUT - 8, y_of[a] - 4, x1, y_of[b] + LH - 2, label)
    img.save(out)
    print("wrote", out.relative_to(ROOT))


def tag(d, x0, y0, x1, y1, label):
    """Red box with the label in a red tag just right of its top-right corner."""
    d.rectangle([x0, y0, x1, y1], outline=RED, width=3)
    tw = int(SANS.getlength(label)) + 16
    tx, ty = (x1, y0) if x1 + tw <= d.im.size[0] else (x1 - tw, y1)   # no room on the right: tag under the box
    d.rectangle([tx, ty, tx + tw, ty + 26], fill=RED)
    d.text((tx + 8, ty + 2), label, font=SANS, fill=(255, 255, 255))


def file_rows(path, segments):
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    rows = []
    for k, (a, b) in enumerate(segments):
        if k:
            rows.append((None, None))
        rows += [(n, lines[n - 1]) for n in range(a, b + 1)]
    return rows


def box(src, dst, *spec):
    """Red boxes on an existing screenshot: groups of x0 y0 x1 y1 label."""
    img = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(img)
    for i in range(0, len(spec), 5):
        x0, y0, x1, y1 = map(int, spec[i:i + 4])
        tag(d, x0, y0, x1, y1, spec[i + 4])
    img.save(dst)


def main():
    p = "firmware/loadcells/hx711_monitor.ino"
    render(p, file_rows(p, [(18, 23), (31, 31), (34, 41), (54, 78)]),
           [(19, 23, "Calibration factors (raw counts per kg)"), (57, 77, "Prints DATA,<kg>,<kg>,<kg> every 250 ms")],
           OUT / "app_06_firmware.png")

    p = "pipeline.py"
    render(p, file_rows(p, [(70, 73), (80, 81), (89, 98)]),
           [(73, 73, "Reject when the tolerance check fails"), (98, 98, "Majority vote, tie goes to the worse grade")],
           OUT / "app_08_grading_logic.png")

    nb = json.loads((ROOT / "crack_pretrain_unet.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if "TEST ({BACKBONE})" in "".join(c["source"]))
    src = "".join(cell["source"]).splitlines()
    i = next(k for k, s in enumerate(src) if s.startswith("model.load_state_dict"))
    out_txt = [l for o in cell["outputs"] if o.get("name") == "stdout" for l in "".join(o["text"]).splitlines() if l]
    rows = [(n + 1, src[n]) for n in range(i, i + 4)] + [(None, ""), (None, "Out:")] + [(f"o{k}", l) for k, l in enumerate(out_txt)]
    render("crack_pretrain_unet.ipynb  (U-Net test cell, DeepCrack test split)", rows,
           [("o0", "o0", "Test IoU, Dice, Precision, Recall")], OUT / "app_05_unet_test.png")

    want = ("torch", "torchvision", "ultralytics", "segmentation_models_pytorch", "segmentation-models-pytorch",
            "opencv-python", "opencv-python-headless", "opencv-contrib-python", "numpy", "scipy", "scikit-image",
            "pyserial", "pillow")
    pl = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout.splitlines()
    py = subprocess.run([sys.executable, "--version"], capture_output=True, text=True).stdout.strip()
    lines = ["> python --version", py, "", "> pip list", *pl[:2], *[l for l in pl[2:] if l.split()[0].lower() in want]]
    render("Terminal  (project venv)", [(None, l) for l in lines], [], OUT / "app_09_pip_list.png", dark=True)


if __name__ == "__main__":
    box(*sys.argv[2:]) if sys.argv[1:2] == ["box"] else main()
