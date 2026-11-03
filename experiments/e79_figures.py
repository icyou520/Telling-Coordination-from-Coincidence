"""Figures for e79 (individual stocks in eight markets).

    uv run python -m experiments.e79_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def markets(res, out):
    from experiments.e79_stocks import COMPARATOR
    inc = [m for m in res["markets"] if m["included"]]
    port = {p["set"]: p for p in res["portfolios"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    for ax, v in zip(axes, ("mean", "breadth")):
        x = np.arange(len(inc))
        s = np.array([m[v]["S"] for m in inc])
        err = np.array([[m[v]["S"] - m[v]["lo"] for m in inc], [m[v]["hi"] - m[v]["S"] for m in inc]])
        ax.errorbar(x - 0.12, s, yerr=err, fmt="o", color=BLUE, ms=6, capsize=3, label="stocks, measured [90%]")
        ax.plot(x - 0.12, [m[v]["S_pred"] for m in inc], "_", color="black", ms=14, mew=2, label="memory gap")
        ps = [port[COMPARATOR[m["market"]]][v]["S"] if m["market"] in COMPARATOR else np.nan for m in inc]
        ax.plot(x + 0.12, ps, "s", color=ORANGE, ms=6, label="portfolios of the region")
        for m, xi in zip(inc, x):
            ax.plot([xi + 0.3] * len(m["virtual"][v]), m["virtual"][v], ".", color=GREY, ms=4)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, [f"{m['market']}\n{m['members']}" for m in inc], fontsize=8)
        ax.set_title(f"Interaction share of the market's {v} log volatility", color=NAVY, loc="left", fontsize=10)
    axes[0].set_ylabel("S (grey dots: virtual groups)")
    axes[0].legend(fontsize=7.5)
    _save(fig, out, "02_markets.png")


def against_gap(res, out):
    fig, ax = plt.subplots(figsize=(6, 5.4), layout="constrained")
    for s in res["sims"]:
        for g, c in (("stocks", BLUE), ("portfolios", ORANGE)):
            ax.plot(s[g]["mean"]["S_pred"], s[g]["mean"]["S"], "o", color=c, ms=4, alpha=0.35)
    for m in res["markets"]:
        if m["included"]:
            ax.plot(m["mean"]["S_pred"], m["mean"]["S"], "o", color=BLUE, ms=8, mec="black")
            ax.annotate(m["market"], (m["mean"]["S_pred"], m["mean"]["S"]), fontsize=7, xytext=(4, 2),
                        textcoords="offset points")
    for p in res["portfolios"]:
        ax.plot(p["mean"]["S_pred"], p["mean"]["S"], "s", color=ORANGE, ms=8, mec="black")
    lim = ax.get_xlim()
    ax.plot(lim, lim, color="black", lw=0.7)
    ax.set_xlabel("memory gap S_pred (no fitting)")
    ax.set_ylabel("measured S (mean log volatility)")
    ax.set_title("Measured against predicted (faint: simulations)\nblue = stocks, orange = portfolios",
                 color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_against_gap.png")


def make_figures(res, out, text=None):
    from experiments.e79_stocks import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e79: pre-registered predictions", out)
    markets(res, out)
    against_gap(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e79_stocks_raw.json").read_text()), Path("results/e79_figures"))
