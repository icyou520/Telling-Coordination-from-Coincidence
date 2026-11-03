"""Figures for e66 (how much of market volatility is coordination between its parts?).

    uv run python -m experiments.e66_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def scorecard(res, out):
    from experiments.e66_market_volatility import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e66: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def shares(res, out):
    from experiments.e66_market_volatility import SIM_CONDITIONS
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), layout="constrained", gridspec_kw={"width_ratios": [4, 2]})
    ax = axes[0]
    for k, c in enumerate(SIM_CONDITIONS):
        rs = [r for r in res["sim"] if r["cond"] == c]
        ax.errorbar(k + np.linspace(-0.2, 0.2, len(rs)), [r["S"] for r in rs],
                    yerr=[[r["S"] - r["lo"] for r in rs], [r["hi"] - r["S"] for r in rs]], fmt="o",
                    color=ORANGE if c == "spillover" else BLUE, ms=4)
        v = [x for r in rs for x in r["virtual"]]
        ax.plot(k + np.linspace(-0.2, 0.2, len(v)), v, "x", color=GREY, ms=4)
    ax.set_xticks(range(len(SIM_CONDITIONS)), ["independent", "correlated\nshocks", "spillover", "common\ndriver"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("interaction share S")
    ax.set_title("Simulated markets (x = virtual groups)", color=NAVY, loc="left")
    ax = axes[1]
    for k, r in enumerate(res["real"]):
        ax.errorbar([k], [r["S"]], yerr=[[r["S"] - r["lo"]], [r["hi"] - r["S"]]], fmt="o", color=NAVY, ms=7)
        ax.plot(k + np.linspace(-0.15, 0.15, len(r["virtual"])), r["virtual"], "x", color=GREY)
    ax.set_xticks(range(len(res["real"])), [f"{r['name']}\n({r['members']})" for r in res["real"]])
    ax.set_xlim(-0.5, len(res["real"]) - 0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Real markets", color=NAVY, loc="left")
    _save(fig, out, "02_shares.png")


def years(res, out):
    ind = next((r for r in res["real"] if r["name"] == "industries"), None)
    if ind is None:
        return
    ys = sorted(ind["years"], key=int)
    s = [ind["years"][y]["S"] for y in ys]
    phi = [ind["years"][y]["phi"] for y in ys]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), layout="constrained", gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axes[0]
    ax.bar([int(y) for y in ys], s, color=BLUE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("interaction share S (that year)")
    ax2 = ax.twinx()
    ax2.plot([int(y) for y in ys], phi, color=ORANGE)
    ax2.set_ylabel("mean |daily return| of industries (%)", color=ORANGE)
    ax.set_title("Industries, year by year", color=NAVY, loc="left")
    axes[1].plot(phi, s, "o", color=BLUE, ms=4)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("mean |daily return| (%)")
    axes[1].set_ylabel("S")
    axes[1].set_title("Turbulence against coordination", color=NAVY, loc="left")
    _save(fig, out, "03_years.png")


def make_figures(res, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(res, out)
    shares(res, out)
    years(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    raw = json.loads(Path("results/e66_market_volatility_raw.json").read_text())
    make_figures(raw, Path("results/e66_figures"))
