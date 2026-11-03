"""Figures for e68 (the memory-gap equation on fish schools).

    uv run python -m experiments.e68_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def scorecard(res, out):
    from experiments.e68_fish_memory_gap import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e68: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def comparison(res, out):
    from experiments.e68_fish_memory_gap import table
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    style = {("adult", "real"): (BLUE, "o", "adult fish, real"), ("adult", "virtual"): (BLUE, "x", "adult fish, virtual"),
             ("zebrafish", "real"): (ORANGE, "s", "zebrafish, real"),
             ("zebrafish", "virtual"): (ORANGE, "+", "zebrafish, virtual")}
    rows = table(res)
    for key, (c, m, lab) in style.items():
        sel = [r for r in rows if (r[0], r[2]) == key]
        ax.plot([r[3] for r in sel], [r[4] for r in sel], m, color=c, ms=7, label=lab)
    ax.axhline(0, color=GREY, lw=0.8), ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("predicted by the memory-gap equation (no twin)")
    ax.set_ylabel("measured with the fitted twin (e33, e38)")
    ax.set_title("Fish schools: equation against full method", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "02_comparison.png")


def make_figures(res, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(res, out)
    comparison(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    raw = json.loads(Path("results/e68_fish_memory_gap_raw.json").read_text())
    make_figures(raw, Path("results/e68_figures"))
