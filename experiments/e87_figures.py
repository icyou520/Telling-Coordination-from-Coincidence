"""Figures for e87 (RANDU).

    uv run python -m experiments.e87_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard

COLORS = {"pcg64": GREY, "minstd": NAVY, "randu": ORANGE, "leaky": BLUE}


def two_tools(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    gens = list(dict.fromkeys(r["gen"] for r in rows))
    for i, g in enumerate(gens):
        rs = [r for r in rows if r["gen"] == g]
        x = i + np.linspace(-0.15, 0.15, len(rs))
        s = np.array([r["S"] for r in rs])
        lo, hi = np.array([r["interval"][0] for r in rs]), np.array([r["interval"][1] for r in rs])
        axes[0].errorbar(x, s, yerr=[s - lo, hi - s], fmt="o", color=COLORS[g], ms=4, capsize=2)
        axes[0].plot(x, [r["null_bar"] for r in rs], "_", color="black", ms=8)
        lp = np.array([-r["log10p"] for r in rs])
        axes[1].scatter(x, np.minimum(lp, 300), color=COLORS[g], s=25)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("interaction share S (black: no-coordination bar)")
    axes[0].set_title("The share: sees the leak, blind to RANDU", color=NAVY, loc="left", fontsize=10)
    axes[1].axhline(3, color="black", lw=0.6, ls="--")
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_ylabel("triple test: -log10 p (capped at 300)")
    axes[1].set_title("3D uniformity test: catches RANDU, blind to the leak", color=NAVY, loc="left", fontsize=10)
    for ax in axes:
        ax.set_xticks(range(len(gens)), gens)
    _save(fig, out, "02_two_tools.png")


def make_figures(rows, out, text=None):
    from experiments.e87_randu import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(rows), "e87: pre-registered predictions", out)
    two_tools(rows, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e87_randu_raw.json").read_text()), Path("results/e87_figures"))
