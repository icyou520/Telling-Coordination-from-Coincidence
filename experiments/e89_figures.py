"""Figures for e89 (time reversal).

    uv run python -m experiments.e89_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def reversal(res, out):
    names = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    y = np.arange(len(names))
    for way, c, dy in (("forward", BLUE, -0.15), ("reversed", ORANGE, 0.15)):
        s = [res[k][way]["S"] for k in names]
        lo = [res[k][way]["interval"][0] for k in names]
        hi = [res[k][way]["interval"][1] for k in names]
        axes[0].errorbar(s, y + dy, xerr=[np.subtract(s, lo), np.subtract(hi, s)], fmt="o", color=c, capsize=2,
                         label=way)
        axes[1].barh(y + dy, [max(res[k][way]["influence"]) for k in names], height=0.3, color=c, label=way)
    for ax in axes:
        ax.set_yticks(y, names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.6)
        ax.legend(fontsize=8)
    axes[0].set_xlabel("interaction share S")
    axes[0].set_title("The share is the same both ways", color=NAVY, loc="left", fontsize=10)
    axes[1].set_xlabel("largest member influence (closure gain)")
    axes[1].set_title("Influence points at the leader only forwards", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "02_forward_vs_reversed.png")


def make_figures(res, out, text=None):
    from experiments.e89_time_reversal import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e89: pre-registered predictions", out)
    reversal(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e89_time_reversal_raw.json").read_text()), Path("results/e89_figures"))
