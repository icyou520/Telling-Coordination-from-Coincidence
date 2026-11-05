"""Figures for e90 (Perth petrol: price leadership and tacit collusion).

    uv run python -m experiments.e90_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def influence(res, out):
    real = {k: r for k, r in res.items() if k.startswith("Perth")}
    sims = {k: r for k, r in res.items() if k.startswith("sim") and k.endswith(sorted(
        {k.rsplit(" ", 1)[1] for k in res if k.startswith("sim")})[0])}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, group, title in ((axes[0], sims, "Simulated markets (first seed): leader vs largest other"),
                             (axes[1], real, "Perth: BP vs largest other brand")):
        names = list(group)
        y = np.arange(len(names))
        lead, other = [], []
        for k in names:
            r = group[k]
            i = r["members"].index("BP") if "members" in r else 0
            inf = np.array(r["influence"])
            lead.append(inf[i])
            other.append(np.delete(inf, i).max())
        ax.barh(y - 0.18, lead, height=0.34, color=ORANGE, label="BP / leader (firm 0)")
        ax.barh(y + 0.18, other, height=0.34, color=BLUE, label="largest other")
        vmax = [max(v["influence_max"] for v in group[k]["virtual"]) for k in names]
        ax.scatter(vmax, y, marker="x", color=GREY, zorder=3, label="virtual groups (max)")
        ax.set_yticks(y, [k.replace("Perth ", "").replace("sim ", "").split(", seed")[0] for k in names], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("influence (closure gain from one member's history)")
        ax.set_title(title, color=NAVY, loc="left", fontsize=10)
        ax.legend(fontsize=8)
    _save(fig, out, "02_influence.png")


def make_figures(res, out, text=None):
    from experiments.e90_petrol import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e90: pre-registered predictions", out)
    influence(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e90_petrol_raw.json").read_text()), Path("results/e90_figures"))
