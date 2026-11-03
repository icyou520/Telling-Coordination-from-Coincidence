"""Figures for e72 (quadratic memory gap for polarization).

    uv run python -m experiments.e72_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def comparison(res, out):
    from experiments.e72_polarization_gap import table
    rows = table(res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained", sharey=True)
    for ax, key, title in ((axes[0], "lin", "Linear memory gap (e68)"), (axes[1], "q", "Quadratic memory gap (e72)")):
        for s, c in (("adult", BLUE), ("zebrafish", ORANGE)):
            for kind, mk in (("real", "o"), ("virtual", "x")):
                pts = [(r[5] if key == "lin" else r[3]["S_q"], r[4]) for r in rows if r[0] == s and r[2] == kind]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=c, ms=6,
                        label=f"{s}, {kind}")
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
        ax.axhline(0, color="#ccc", lw=0.6)
        ax.axvline(0, color="#ccc", lw=0.6)
        ax.set_xlabel("predicted from memory (no fish model)")
        ax.set_title(title, color=NAVY, loc="left", fontsize=10)
    axes[0].set_ylabel("measured S_proj (fitted fish twins)")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out, "02_comparison.png")


def make_figures(res, out, text=None):
    from experiments.e72_polarization_gap import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e72: pre-registered predictions", out)
    comparison(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e72_polarization_gap_raw.json").read_text()), Path("results/e72_figures"))
