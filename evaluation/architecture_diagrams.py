"""
architecture_diagrams.py - figures of D1/D2/D3 with the proposed classifier head, for comparing in Ch.3.

Head, same for every design (designs.head()): pooled features -> FC 128 + ReLU -> Dropout 0.2 -> Linear 3.
It replaces only the last Linear layer of each design; the backbones are unchanged.

    python evaluation/architecture_diagrams.py
Outputs in evaluation/results/figures/: arch_d1.{png,svg}, arch_d2.{png,svg}, arch_d3.{png,svg}
"""
import math
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

from designs import HIDDEN, IMGSZ, build

OUT = os.path.join(os.path.dirname(__file__), "results", "figures")
COLOR = dict(stem="#4a86c8", res="#f2a44b", pool="#8d72c9", out="#d1493c", new="#3fa560", removed="#f2f2f2")
LEGEND = [("stem", "Stem convolution"), ("res", "Feature stage"), ("pool", "Pooling / existing FC"),
          ("new", "Added: proposed head"), ("out", "Output")]


def head_of(model, design):
    return model.fc if design == "d3" else model.classifier[-1]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# (name, sublabel, channels, spatial px, kind); spatial 1 = vector, kind "removed" = dashed box
SPECS = dict(
    d1=("A. MobileNetV3-Large-Based Grade Classification Architecture + Proposed Head", [
        ("Input patch", "224 × 224 px\nRGB", 3, 224, "input"),
        ("Conv 3×3", "stride 2, h-swish", 16, 112, "stem"),
        ("Bneck 1", "1 block, 3×3", 16, 112, "res"),
        ("Bneck 2", "2 blocks, 3×3", 24, 56, "res"),
        ("Bneck 3", "3 blocks, 5×5, SE", 40, 28, "res"),
        ("Bneck 4", "4 blocks, 3×3", 80, 14, "res"),
        ("Bneck 5", "2 blocks, 3×3, SE", 112, 14, "res"),
        ("Bneck 6", "3 blocks, 5×5, SE", 160, 7, "res"),
        ("Conv 1×1", "h-swish", 960, 7, "stem"),
        ("Avg pool", "7×7 → 1×1", 960, 1, "pool"),
        ("FC 1280", "h-swish, dropout 0.2", 1280, 1, "pool"),
        ("FC 128 + Dropout", "ReLU, p = 0.2", HIDDEN, 1, "new"),
        ("Linear", "softmax", 3, 1, "out"),
    ]),
    d2=("B. Reduced EfficientNet-B0-Based Grade Classification Architecture + Proposed Head", [
        ("Input patch", "160 × 160 px\nRGB", 3, 160, "input"),
        ("Conv 3×3", "stride 2, SiLU", 32, 80, "stem"),
        ("Stage 1", "1 MBConv1, 3×3", 16, 80, "res"),
        ("Stage 2", "2 MBConv6, 3×3", 24, 40, "res"),
        ("Stage 3", "2 MBConv6, 5×5", 40, 20, "res"),
        ("Stage 4", "3 MBConv6, 3×3", 80, 10, "res"),
        ("Stage 5", "3 MBConv6, 5×5", 112, 10, "res"),
        ("Stage 6", "4 MBConv6, 5×5", 192, 5, "res"),
        ("Stage 7", "1 MBConv6, 3×3", 320, 5, "removed"),
        ("Conv 1×1", "1280-ch head", 1280, 5, "removed"),
        ("Avg pool", "5×5 → 1×1, dropout 0.2", 192, 1, "pool"),
        ("FC 128 + Dropout", "ReLU, p = 0.2", HIDDEN, 1, "new"),
        ("Linear", "softmax", 3, 1, "out"),
    ]),
    d3=("C. Reduced ResNet18-Based Grade Classification Architecture + Proposed Head", [
        ("Input patch", "224 × 224 px\nRGB", 3, 224, "input"),
        ("Conv 7×7", "stride 2, ReLU", 64, 112, "stem"),
        ("Max pool", "3×3, stride 2", 64, 56, "pool"),
        ("Layer 1", "2 basic blocks", 64, 56, "res"),
        ("Layer 2", "2 basic blocks", 128, 28, "res"),
        ("Layer 3", "2 basic blocks", 256, 14, "res"),
        ("Layer 4", "2 basic blocks", 512, 7, "removed"),
        ("Avg pool", "14×14 → 1×1", 256, 1, "pool"),
        ("FC 128 + Dropout", "ReLU, p = 0.2", HIDDEN, 1, "new"),
        ("Linear", "softmax", 3, 1, "out"),
    ]),
)
BLOCK = dict(d1="Bneck: inverted residual (1×1 expand, depthwise k×k, 1×1 project), SE = squeeze-excitation.",
             d2="MBConvN: inverted residual with expansion N and squeeze-excitation.",
             d3="Basic block: two 3×3 convolutions.")


def shade(hex_color, f):
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(np.clip(rgb * f if f < 1 else rgb + (1 - rgb) * (f - 1), 0, 1))


def box(ax, x, h, d, color, dashed=False):
    """3D block: front face d wide, h tall, centred on y=0; top and side faces receding up-right by o."""
    o, y0, y1 = 0.35 * h, -h / 2, h / 2
    ls, ec = ("--", "#9a9a9a") if dashed else ("-", "#555555")
    faces = [([(x, y0), (x + d, y0), (x + d, y1), (x, y1)], shade(color, 0.92)),
             ([(x, y1), (x + d, y1), (x + d + o, y1 + o), (x + o, y1 + o)], shade(color, 1.35)),
             ([(x + d, y0), (x + d + o, y0 + o), (x + d + o, y1 + o), (x + d, y1)], color)]
    for pts, fc in faces:
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.6, ls=ls, zorder=3))
    return x + d + o, y1 + o


def vector(ax, x, length, color):
    """1×1 feature vector: thin bar rising diagonally."""
    t, dx = 0.12, length * 0.75
    ax.add_patch(Polygon([(x, -t), (x + dx, length * 0.5 - t), (x + dx, length * 0.5 + t), (x, t)],
                         closed=True, fc=color, ec="#555555", lw=0.6, zorder=3))
    return x + dx, length * 0.5 + t


def draw(design, path):
    title, layers = SPECS[design]
    model = build(design, pretrained=False)
    head = head_of(model, design)
    new = n_params(model)
    base = new - n_params(head) + (head[0].in_features + 1) * head[-1].out_features   # with the old single Linear
    fig, ax = plt.subplots(figsize=(0.95 * len(layers) + 1.5, 3.9))
    ax.set_axis_off()
    x, prev_right, gap = 0.0, None, 0.55
    log_sp = lambda s: (math.log(s) - math.log(5)) / (math.log(112) - math.log(5))
    for i, (name, sub, ch, sp, kind) in enumerate(layers):
        if prev_right is not None:
            ax.add_patch(FancyArrowPatch((prev_right, 0), (x - 0.04, 0), arrowstyle="-|>", mutation_scale=7,
                                         color="#2f5f5f", lw=1, zorder=2))
        if kind == "input":
            s = 2.3
            noise = np.random.default_rng(0).random((48, 48))
            ax.imshow(noise, cmap="gray", extent=(x, x + s * 0.6, -s / 2, s / 2), zorder=3, aspect="auto")
            right, top, mid = x + s * 0.6, s / 2, x + s * 0.3
        elif sp == 1:
            right, top = vector(ax, x, 0.3 + 0.18 * math.log2(ch / 3), COLOR[kind])
            mid = (x + right) / 2
            ax.text(right + 0.05, top, str(ch), fontsize=6.5, va="bottom", color="#333333")
        else:
            h, d = 0.55 + 1.75 * log_sp(sp), 0.12 + 0.11 * math.log2(ch / 8)
            right, top = box(ax, x, h, d, COLOR["removed" if kind == "removed" else kind], kind == "removed")
            mid = x + d / 2
            grey = "#9a9a9a" if kind == "removed" else "#333333"
            ax.text(x, -h / 2 - 0.08, str(ch), fontsize=6.5, ha="left", va="top", color=grey)
            ax.text(right + 0.03, 0.35 * h / 2, f"{sp}×{sp}", fontsize=5.5, rotation=90, ha="left", va="center",
                    color=grey)
            if kind == "res":
                ax.add_patch(FancyArrowPatch((x - 0.1, h / 2 + 0.05), (right + 0.05, top + 0.08),
                                             connectionstyle="arc3,rad=-0.55", arrowstyle="-|>", mutation_scale=6,
                                             color="#b0672a", lw=0.7, ls="--", zorder=4))
        if kind in ("removed", "new"):
            ax.text(mid + 0.15, top + 0.25, "removed" if kind == "removed" else "added", fontsize=7, ha="center",
                    style="italic", color="#c0392b" if kind == "removed" else "#2e8b4e")
            ax.plot([mid - 0.25, mid + 0.55], [top + 0.2] * 2, color="#c0392b" if kind == "removed" else "#2e8b4e",
                    lw=0.8)
        row = -1.85 - 0.75 * (i % 2)
        faded = kind == "removed"
        ax.text(mid, row, name, fontsize=8, ha="center", va="top", weight="bold", style="italic" if faded else None,
                color="#9a9a9a" if faded else "#111111")
        ax.text(mid, row - 0.24, sub, fontsize=6.5, ha="center", va="top", color="#9a9a9a" if faded else "#333333")
        pad = 0.18 if kind != "input" and sp > 1 else 0   # room for the rotated spatial label
        prev_right, x = right + pad, right + pad + gap
    ax.text(x - 0.25, 0.25, "Grade A\nGrade B\nGrade C", fontsize=7.5, va="center", linespacing=1.3)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLOR[k], ec="#555555", lw=0.5) for k, _ in LEGEND]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=COLOR["removed"], ec="#9a9a9a", ls="--", lw=0.7))
    labels = [l for _, l in LEGEND] + ["Removed (slim design)"]
    if design == "d1":
        handles, labels = handles[:-1], labels[:-1]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=len(labels), fontsize=6.5,
              frameon=False, handlelength=1.2, columnspacing=1.2)
    fig.text(0.01, 0.015, "Number under each layer: output channels. Number along each layer: spatial size in px. "
             f"{BLOCK[design]} {base / 1e6:.2f}M → {new / 1e6:.2f}M parameters ({(new - base) / 1e3:+.1f}k from head).",
             fontsize=6, color="#444444")
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.set_xlim(-0.2, x + 0.6)
    ax.set_ylim(-3.3, 2.6)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12)
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    plt.close(fig)
    return base, new


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for d in SPECS:
        s = IMGSZ[d]
        with torch.no_grad():
            assert build(d, pretrained=False).eval()(torch.zeros(1, 3, s, s)).shape == (1, 3)
        base, new = draw(d, os.path.join(OUT, f"arch_{d}"))
        print(f"{d}: {base:,} -> {new:,} params ({new - base:+,})")
