"""Figures for e88 (group size).

    uv run python -m experiments.e88_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard

COLORS = {"mean-field": BLUE, "topological": ORANGE, "metric": NAVY}


def size(rows, out):
    from experiments.e88_group_size import theory
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for rule, c in COLORS.items():
        rs = [r for r in rows if r["rule"] == rule]
        n = [r["N"] for r in rs]
        axes[0].plot(n, [r["S"] for r in rs], "o-", color=c, label=rule)
        axes[0].plot(n, [r["S_virtual"] for r in rs], "x", color=c)
        axes[1].plot(n, [r["rho_member"] for r in rs], "o-", color=c, label=f"{rule}: members")
        axes[1].plot(n, [r["rho_group"] for r in rs], "s--", color=c, ms=3, alpha=0.6)
    nn = np.unique(np.geomspace(2, max(r["N"] for r in rows), 60).astype(int))
    axes[0].plot(nn, [theory(n) for n in nn], color=GREY, lw=3, alpha=0.5, zorder=0, label="mean-field theory")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("interaction share S (memory gap)")
    axes[0].set_title("The share saturates with group size (crosses: virtual groups)", color=NAVY, loc="left",
                      fontsize=10)
    axes[1].set_ylabel("lag-1 autocorrelation")
    axes[1].set_title("Group memory is fixed (dashed); members' memory falls", color=NAVY, loc="left", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("group size N")
        ax.legend(fontsize=8)
    _save(fig, out, "02_share_vs_size.png")


def make_figures(rows, out, text=None):
    from experiments.e88_group_size import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(rows), "e88: pre-registered predictions", out)
    size(rows, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e88_group_size_raw.json").read_text()), Path("results/e88_figures"))
