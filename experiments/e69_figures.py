"""Figures for e69 (log market volatility on new data).

    uv run python -m experiments.e69_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def scorecard(res, out):
    from experiments.e69_market_log_volatility import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e69: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def shares(res, out):
    from experiments.e69_market_log_volatility import SETS
    labels = {"holdout": "1936-1971\n(new data)", "equal": "equal-weighted\n1972-2026", "reference": "1972-2026\n(seen before)"}
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    for k, name in enumerate(SETS):
        for dx, kind, c in ((-0.15, "log", BLUE), (0.15, "raw", ORANGE)):
            r = res[name][kind]
            ax.errorbar([k + dx], [r["S"]], yerr=[[r["S"] - r["lo"]], [r["hi"] - r["S"]]], fmt="o", color=c, ms=7,
                        label=f"{kind} volatility, measured" if k == 0 else None)
            ax.plot([k + dx], [r["S_pred"]], "D", mfc="none", color=c, ms=9,
                    label=f"{kind} volatility, equation" if k == 0 else None)
            ax.plot(np.full(len(r["virtual"]), k + dx + 0.06), r["virtual"], "x", color=GREY, ms=4)
    ax.set_xticks(range(len(SETS)), [labels[n] for n in SETS])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("interaction share of market volatility")
    ax.set_title("Log volatility: small but real?", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "02_shares.png")


def make_figures(res, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(res, out)
    shares(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    raw = json.loads(Path("results/e69_market_log_volatility_raw.json").read_text())
    make_figures(raw, Path("results/e69_figures"))
