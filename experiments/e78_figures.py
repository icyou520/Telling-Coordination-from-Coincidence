"""Figures for e78 (segment rotation audit of e51-e60).

    uv run python -m experiments.e78_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard

COLS = ("published", "first", "middle")


def verdicts(res, out):
    from experiments.e78_split_audit import REAL
    keys = [(k, p) for k, ps in REAL.items() for p in ps]
    grid = np.array([[res["published"][k][p]["ok"]] + [res["rotations"][s]["verdicts"][k][p]["ok"]
                                                        for s in COLS[1:]] for k, p in keys], float)
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(keys) + 1.4), layout="constrained")
    ax.imshow(grid, cmap=plt.matplotlib.colors.ListedColormap(["#e8c4c0", "#bfe0c4"]), vmin=0, vmax=1,
              aspect="auto")
    for i in range(len(keys)):
        for j in range(3):
            ax.text(j, i, "PASS" if grid[i, j] else "fail", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(3), ["published\n(last 30%)", "first 30%", "middle 30%"], fontsize=8)
    ax.set_yticks(range(len(keys)), [f"{k} {p}" for k, p in keys], fontsize=7.5)
    ax.set_title("Real-data verdicts by test segment", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "02_verdicts.png")


def e51_values(res, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    sets = [res["published_e51"]] + [res["rotations"][s]["e51"] for s in COLS[1:]]
    for ax, t in zip(axes, ("turn", "shape", "motion")):
        x = np.arange(len(sets[0][t]))
        for k, (vals, c, lab) in enumerate(zip(sets, (GREY, BLUE, ORANGE), COLS)):
            ax.plot(x + (k - 1) * 0.15, vals[t], "o", color=c, ms=6, label=lab)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, [f"rec {i}" for i in x], fontsize=7)
        ax.set_title(f"e51 G({t}), 8-fish schools", color=NAVY, loc="left", fontsize=10)
    axes[0].set_ylabel("G (share of turn error removed)")
    axes[0].legend(fontsize=7.5)
    _save(fig, out, "03_e51_values.png")


def make_figures(res, out, text=None):
    from experiments.e78_split_audit import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e78: pre-registered predictions", out)
    verdicts(res, out)
    e51_values(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e78_split_audit_raw.json").read_text()), Path("results/e78_figures"))
