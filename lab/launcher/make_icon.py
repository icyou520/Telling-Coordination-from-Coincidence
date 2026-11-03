"""Draw the Emergence Lab app icon: a school of fish on navy, its fast leader at the front in orange.

    uv run python -m lab.launcher.make_icon OUT.png
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

TOP, BOTTOM, ORANGE, TEAL, WHITE = "#2b538f", "#0d1a33", "#f0922f", "#2cc5b8", "#eef2fa"


def fish(ax, x, y, ang, size, color, alpha=1.0, z=3):
    t = np.linspace(-1, 1, 40)
    bx = 0.5 + 0.5 * t                                   # nose at 1, tail joint at 0
    half = 0.17 * np.sqrt(np.clip(1 - t ** 2, 0, None)) * (1.15 - 0.3 * t)
    body = np.concatenate([np.column_stack([bx, half]), np.column_stack([bx[::-1], -half[::-1]])])
    tail = np.array([[0.05, 0.0], [-0.32, 0.22], [-0.22, 0.0], [-0.32, -0.22]])
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, s], [-s, c]])
    for shape in (body, tail):
        ax.add_patch(Polygon((shape - [0.35, 0]) * size @ rot + [x, y], closed=True, color=color, alpha=alpha,
                             lw=0, zorder=z))


def draw(out: str):
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_alpha(0)
    m = 0.06
    card = FancyBboxPatch((m, m), 1 - 2 * m, 1 - 2 * m, boxstyle="round,pad=0,rounding_size=0.2",
                          facecolor="none", lw=0)
    ax.add_patch(card)
    yy, xx = np.mgrid[0:1:256j, 0:1:256j]
    im = ax.imshow(0.75 * yy + 0.25 * xx, extent=(m, 1 - m, m, 1 - m), origin="lower", zorder=0,
                   cmap=LinearSegmentedColormap.from_list("g", [BOTTOM, TOP]))
    im.set_clip_path(card)

    for r, a, lw in ((0.31, 0.35, 7), (0.375, 0.18, 5)):
        t = np.linspace(-0.62 * np.pi, 0.08 * np.pi, 200)
        line, = ax.plot(0.5 + r * np.cos(t), 0.5 + r * np.sin(t), color=TEAL, lw=lw, alpha=a,
                        solid_capstyle="round", zorder=1)
        t2 = np.linspace(0.3 * np.pi, 1.2 * np.pi, 200)
        ax.plot(0.5 + r * np.cos(t2), 0.5 + r * np.sin(t2), color=TEAL, lw=lw, alpha=a * 0.6,
                solid_capstyle="round", zorder=1)

    rng = np.random.default_rng(7)
    heading = np.deg2rad(32)
    fwd, left = np.array([np.cos(heading), np.sin(heading)]), np.array([-np.sin(heading), np.cos(heading)])
    centre = np.array([0.43, 0.43])
    pts = []
    while len(pts) < 17:
        u, v = rng.uniform(-1, 1, 2)
        if u * u + v * v > 1:
            continue
        p = centre + 0.2 * u * fwd + 0.15 * v * left
        if all(np.hypot(*(p - q)) > 0.072 for q in pts):
            pts.append(p)
    for p in pts:
        lat = (p - centre) @ left
        fish(ax, *p, heading - 1.2 * lat + rng.normal(0, 0.06), 0.08, WHITE, 0.6 + 0.35 * rng.random())
    lead = centre + 0.40 * fwd + 0.03 * left
    for i, k in enumerate((0.075, 0.12, 0.16)):
        ax.plot(*(lead - k * fwd), "o", color=ORANGE, ms=13 - 3 * i, alpha=0.55 - 0.15 * i, zorder=2)
    fish(ax, *lead, heading + 0.25, 0.14, ORANGE, z=4)
    fig.savefig(out, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "icon.png")
