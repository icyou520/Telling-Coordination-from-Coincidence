"""Figures for e91 (Sydney petrol: shared price leadership).

    uv run python -m experiments.e91_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def influence(res, out):
    from experiments.e91_sydney import LINE, MAJORS
    first = sorted({k.rsplit(" ", 1)[1] for k in res if k.startswith("sim")})[0]
    sims = {k: r for k, r in res.items() if k.startswith("sim") and k.endswith(first)}
    real = res.get("Sydney 2016-2017")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    names = list(sims)
    y = np.arange(len(names))
    lead = [sims[k]["influence"][0] for k in names]
    other = [max(sims[k]["influence"][1:]) for k in names]
    ax.barh(y - 0.18, lead, height=0.34, color=ORANGE, label="firm 0 (leader where there is one)")
    ax.barh(y + 0.18, other, height=0.34, color=BLUE, label="largest other")
    ax.scatter([max(v["influence_max"] for v in sims[k]["virtual"]) for k in names], y, marker="x", color=GREY,
               zorder=3, label="virtual groups (max)")
    ax.set_yticks(y, [k.replace("sim ", "").split(", seed")[0] for k in names], fontsize=8)
    ax.set_title("Simulated markets (first seed)", color=NAVY, loc="left", fontsize=10)
    panels = [(ax, None)]
    if not real:
        axes[1].set_visible(False)
    else:
        ax = axes[1]
        inf = np.array(real["influence"])
        order = np.argsort(-inf)
        y = np.arange(len(order))
        ax.barh(y, inf[order], color=[ORANGE if real["members"][i] in MAJORS else BLUE for i in order])
        ax.axvline(max(v["influence_max"] for v in real["virtual"]), color=GREY, lw=0.8, ls=":",
                   label="virtual groups (max)")
        ax.set_yticks(y, [real["members"][i] for i in order], fontsize=8)
        ax.set_title(f"Sydney 2016-2017 (S {real['S']:+.3f}): majors orange", color=NAVY, loc="left", fontsize=10)
        panels.append((ax, None))
    for ax, _ in panels:
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.6)
        ax.axvline(LINE, color=GREY, lw=0.8, ls="--")
        ax.set_xlabel("influence (closure gain from one member's history)")
        ax.legend(fontsize=8)
    _save(fig, out, "02_influence.png")


def make_figures(res, out, text=None):
    from experiments.e91_sydney import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e91: pre-registered predictions", out)
    influence(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e91_sydney_raw.json").read_text()), Path("results/e91_figures"))
