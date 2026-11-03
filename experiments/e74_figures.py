"""Figures for e74 (the memory gap for nonlinear group variables in markets).

    uv run python -m experiments.e74_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, RED, _save, plt, summary_text
from experiments.e71_figures import scorecard

LABELS = {"mean": "mean", "breadth": "breadth", "max": "worst industry", "max_joint": "worst industry,\njoint twin"}
COLORS = {"mean": BLUE, "breadth": ORANGE, "max": RED, "max_joint": GREY}


def excess(res, out):
    from experiments.e74_nonlinear_gap import CONDS, VARS
    groups = [(c, [r for r in res["sim"] if r["cond"] == c]) for c in CONDS]
    groups += [(r["name"], [r]) for r in res["real"]]
    fig, ax = plt.subplots(figsize=(12, 4.6), layout="constrained")
    w = 0.8 / len(VARS)
    for k, v in enumerate(VARS):
        med = [np.median([r[v]["excess"] for r in rs]) for _, rs in groups]
        ax.bar(np.arange(len(groups)) + (k - 1.5) * w, med, w, color=COLORS[v], label=LABELS[v].replace("\n", " "))
    ax.set_yscale("symlog", linthresh=0.02)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(len(CONDS) - 0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9)
    ax.set_ylabel("share minus virtual floor (symlog)")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.set_title("Simulated markets (median of 5 seeds) and real industries", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "02_excess.png")


def additive(res, out):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), layout="constrained")
    for v, c in (("mean", BLUE), ("breadth", ORANGE)):
        rs = res["sim"] + res["real"]
        ax.plot([r[v]["S_pred"] for r in rs], [r[v]["S"] for r in rs], "o", color=c, label=v, ms=5)
    lim = max(0.03, max(max(r[v]["S"], r[v]["S_pred"]) for r in res["sim"] + res["real"] for v in ("mean", "breadth")))
    ax.plot([-0.01, lim], [-0.01, lim], "--", color=GREY, lw=0.8)
    ax.set_xlabel("memory-gap prediction")
    ax.set_ylabel("measured share")
    ax.legend(fontsize=8)
    ax.set_title("Additive group variables", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_additive.png")


def make_figures(res, out, text=None):
    from experiments.e74_nonlinear_gap import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e74: pre-registered predictions", out)
    excess(res, out)
    additive(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e74_nonlinear_gap_raw.json").read_text()), Path("results/e74_figures"))
