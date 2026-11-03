"""Figures for e75 (an exact memory twin for non-additive group variables).

    uv run python -m experiments.e75_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, RED, _save, plt, summary_text
from experiments.e71_figures import scorecard


def shares(res, out):
    from experiments.e75_exact_twin import CONDS
    groups = [(c, [r for r in res["sim"] if r["cond"] == c]) for c in CONDS]
    groups += [(r["name"], [r]) for r in res["real"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    for ax, v, title in ((axes[0], "max", "Worst industry"), (axes[1], "systemic", "Systemic day (>= 20% extreme)")):
        x = np.arange(len(groups))
        s = [np.median([r[v]["S"] for r in rs]) for _, rs in groups]
        j = [np.median([r[v]["joint"] for r in rs]) for _, rs in groups]
        ax.bar(x - 0.2, s, 0.4, color=RED, label="independent twin (S)")
        ax.bar(x + 0.2, j, 0.4, color=GREY, label="joint twin (dynamic part)")
        for k, (_, rs) in enumerate(groups):
            vv = [s for r in rs for s in r["virtual"][v]]
            ax.plot([k + 0.45] * len(vv), vv, "x", color=BLUE, ms=4)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(len(CONDS) - 0.5, color=GREY, lw=0.8, ls=":")
        ax.set_xticks(x, [g[0].replace("e74 ", "e74\n").replace(" 19", "\n19") for g in groups], fontsize=7)
        ax.set_title(f"{title}; x = virtual groups, nothing subtracted", color=NAVY, loc="left", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("interaction share (symlog)")
    _save(fig, out, "02_shares.png")


def bias(res, out):
    fig, ax = plt.subplots(figsize=(6, 4.2), layout="constrained")
    ind = [r for r in res["sim"] if r["cond"] == "independent"]
    old = [r["max"]["e74"] for r in ind]
    new = [r["max"]["S"] for r in ind]
    vir = [s for r in res["sim"] for s in r["virtual"]["max"]]
    for k, (vals, label, c) in enumerate(((old, "e74 twin,\nindependent", ORANGE), (new, "volatility twin,\nindependent", BLUE),
                                          (vir, "volatility twin,\nvirtual groups", GREY))):
        ax.plot(k + np.random.default_rng(k).uniform(-0.12, 0.12, len(vals)), vals, "o", color=c, ms=5)
    ax.set_xticks(range(3), ["e74 twin,\nindependent", "volatility twin,\nindependent", "volatility twin,\nvirtual groups"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("worst-industry share, nothing subtracted")
    ax.set_title("Where the true share is 0", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_bias.png")


def make_figures(res, out, text=None, e74_res=None):
    from experiments.e75_exact_twin import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res, e74_res), "e75: pre-registered predictions", out)
    shares(res, out)
    bias(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    from experiments.e75_exact_twin import _e74
    make_figures(json.loads(Path("results/e75_exact_twin_raw.json").read_text()), Path("results/e75_figures"),
                 e74_res=_e74())
