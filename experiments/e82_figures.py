"""Figures for e82 (the share and next month's correlation between firms, on fresh markets).

    uv run python -m experiments.e82_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, _save, plt, summary_text
from experiments.e71_figures import scorecard


def _bars(ax, mk, a, b, title):
    x = np.arange(len(mk))
    ax.bar(x - 0.2, a, 0.4, color=BLUE, label="share S")
    ax.bar(x + 0.2, b, 0.4, color=GREY, label="placebo S (time-shifted firms)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, [m["market"] for m in mk], fontsize=7, rotation=45, ha="right")
    ax.set_title(title, color=NAVY, loc="left", fontsize=10)


def coefficients(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained", gridspec_kw={"width_ratios": [8, 14]})
    for ax, s, t in ((axes[0], "discovery", "Discovery: e81's markets"), (axes[1], "fresh", "Fresh markets (scored)")):
        mk = [m for m in res[s] if m["included"]]
        _bars(ax, mk, [m["coef_S"] for m in mk], [m["coef_Sv"] for m in mk], t)
    axes[0].set_ylabel("effect on next month's correlation\nper s.d., beyond current correlation")
    axes[0].legend(fontsize=8)
    _save(fig, out, "02_coefficients.png")


def gains(res, out):
    g = lambda m, a, b: 100 * (1 - m["mse"][b] / m["mse"][a])  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained", gridspec_kw={"width_ratios": [8, 14]})
    for ax, s, t in ((axes[0], "discovery", "Discovery: e81's markets"), (axes[1], "fresh", "Fresh markets (scored)")):
        mk = [m for m in res[s] if m["included"]]
        _bars(ax, mk, [g(m, "full", "full+S") for m in mk], [g(m, "full", "full+S_v") for m in mk], t)
        ax.scatter(np.arange(len(mk)) - 0.2, [g(m, "e81", "e81+S") for m in mk], marker="x", color="black", s=18,
                   label="S without correlation controls (e81's test)", zorder=3)
    axes[0].set_ylabel("% lower out-of-sample error,\nbeyond HAR + DY + current correlation")
    axes[0].legend(fontsize=7)
    _save(fig, out, "03_forecast_gains.png")


def make_figures(res, out, text=None):
    from experiments.e82_correlation import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e82: pre-registered predictions (fresh markets)", out)
    coefficients(res, out)
    gains(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e82_correlation_raw.json").read_text()), Path("results/e82_figures"))
