"""Figures for e81 (the interaction share against the Diebold-Yilmaz spillover index).

    uv run python -m experiments.e81_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard

CONDS = ("independent", "correlated", "spillover", "factor")
LABELS = ("independent", "simultaneous\nshocks", "spillover", "common\nfactor")


def known_answers(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key, title, c in ((axes[0], "S", "Interaction share S (memory gap)", BLUE),
                              (axes[1], "DY", "Diebold-Yilmaz total spillover (%)", ORANGE)):
        for i, cond in enumerate(CONDS):
            v = [s[key] for s in res["sim"] if s["cond"] == cond]
            w = [s[key + "_win"] for s in res["sim"] if s["cond"] == cond]
            ax.bar(i - 0.18, np.median(v), 0.34, color=c, label="full sample" if i == 0 else None)
            ax.bar(i + 0.18, np.median(w), 0.34, color=c, alpha=0.4, label="500-day windows (median)" if i == 0 else None)
        ax.set_xticks(range(len(CONDS)), LABELS)
        ax.set_title(title, color=NAVY, loc="left", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Simulated markets with known answers", x=0.01, ha="left", color=NAVY)
    _save(fig, out, "02_known_answers.png")


def over_time(res, out):
    mk = [m for m in res["markets"] if m["included"]]
    fig, axes = plt.subplots(len(mk), 1, figsize=(11, 1.6 * len(mk)), sharex=False, layout="constrained")
    for ax, m in zip(np.atleast_1d(axes), mk):
        w = m["windows"]
        z = lambda v: (np.array(v) - np.mean(v)) / (np.std(v) + 1e-12)  # noqa: E731
        x = np.arange(len(w))
        ax.plot(x, z([r["S"] for r in w]), color=BLUE, lw=1, label="S")
        ax.plot(x, z([r["DY"] for r in w]), color=ORANGE, lw=1, label="DY")
        ax.plot(x, z([r["y_vol"] for r in w]), color=GREY, lw=0.8, label="next month's volatility")
        ax.set_ylabel(m["market"], fontsize=8)
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=3, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("window (stepped 21 trading days from 2000); each series standardized")
    _save(fig, out, "03_over_time.png")


def gains(res, out):
    mk = [m for m in res["markets"] if m["included"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained", sharey=False)
    for ax, t, title in ((axes[0], "y_vol", "next month's market volatility"),
                         (axes[1], "y_corr", "next month's correlation between firms")):
        x = np.arange(len(mk))
        g = lambda b, a: [100 * (1 - m["oos"][t][b] / m["oos"][t][a]) for m in mk]  # noqa: E731
        ax.bar(x - 0.25, g("B", "A"), 0.25, color=BLUE, label="adding S to HAR + DY")
        ax.bar(x, g("B", "C"), 0.25, color=ORANGE, label="adding DY to HAR + S")
        ax.bar(x + 0.25, g("P", "A"), 0.25, color=GREY, label="adding placebo S to HAR + DY")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, [m["market"] for m in mk], fontsize=8)
        ax.set_title(f"Out-of-sample gain in forecasting {title}", color=NAVY, loc="left", fontsize=10)
        ax.set_ylabel("% lower squared error")
    axes[0].legend(fontsize=8)
    _save(fig, out, "04_forecast_gains.png")


def make_figures(res, out, text=None):
    from experiments.e81_spillover import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e81: pre-registered predictions", out)
    known_answers(res, out)
    over_time(res, out)
    gains(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e81_spillover_raw.json").read_text()), Path("results/e81_figures"))
