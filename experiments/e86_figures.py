"""Figures for e86 (the Ising magnet).

    uv run python -m experiments.e86_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard

TC = 2 / np.log(1 + np.sqrt(2))


def share(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = [GREY, BLUE, ORANGE, NAVY]
    for (L, rows), c in zip(res["scans"].items(), colors):
        t = [r["T"] for r in rows]
        axes[0].plot(t, [r["S"] for r in rows], "o-", color=c, ms=3, label=f"L = {L}")
        axes[0].plot(t, [r["S_virtual"] for r in rows], "x", color=c, ms=3)
        axes[1].plot(t, [r["chi"] for r in rows], "o-", color=c, ms=3, label=f"L = {L}")
    for ax in axes:
        ax.axvline(TC, color="black", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([1.6, 2, 2.269, 3, 4, 6, 10], ["1.6", "2", "Tc", "3", "4", "6", "10"])
        ax.set_xlabel("temperature T")
        ax.legend(fontsize=8)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("interaction share S (memory gap)")
    axes[0].set_title("Share of the magnetization (crosses: virtual lattices)", color=NAVY, loc="left", fontsize=10)
    axes[1].set_ylabel("susceptibility")
    axes[1].set_yscale("log")
    axes[1].set_title("For reference: the textbook marker of the transition", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "02_share_vs_temperature.png")


def make_figures(res, out, text=None):
    from experiments.e86_ising import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e86: pre-registered predictions", out)
    share(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e86_ising_raw.json").read_text()), Path("results/e86_figures"))
