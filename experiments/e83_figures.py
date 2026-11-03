"""Figures for e83 (the share of 50-fish schools across light levels).

    uv run python -m experiments.e83_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def dose(res, out):
    real = [r for r in res["real"] if r["used"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    x = np.array([r["light"] for r in real]) * 900
    ax.scatter(x, [r["S"] for r in real], color=BLUE, s=40, label="school (one dot per recording)", zorder=3)
    ax.scatter(x, [r["S_virtual"] for r in real], color=GREY, s=25, marker="x", label="virtual school (tracks shifted)")
    lv = sorted(set(x))
    ax.plot(lv, [np.mean([r["S"] for r, xx in zip(real, x) if xx == v]) for v in lv], color=BLUE, lw=1, alpha=0.6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("light (lux)")
    ax.set_ylabel("interaction share S (memory gap)")
    ax.set_title("Schools of ~50 tetras at fixed light levels", color=NAVY, loc="left", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, out, "02_dose_response.png")


def sim_check(res, out):
    sim = res["sim"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    t, s = np.array([r["true"] for r in sim]), np.array([r["S"] for r in sim])
    axes[0].scatter(t, s, color=ORANGE, s=18)
    lim = [min(t.min(), s.min()) - 0.01, max(t.max(), s.max()) + 0.02]
    axes[0].plot(lim, lim, color="black", lw=0.6)
    axes[0].set_xlabel("share with full identities")
    axes[0].set_ylabel("share from broken tracks")
    axes[0].set_title("Simulated schools: broken tracks recover the share", color=NAVY, loc="left", fontsize=10)
    conds = [c for c in dict.fromkeys(r["cond"] for r in sim)]
    for i, c in enumerate(conds):
        v = [r["S"] for r in sim if r["cond"] == c]
        axes[1].scatter([i] * len(v), v, color=BLUE, s=14)
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_xticks(range(len(conds)), ["loners", "sees 0.05", "0.1", "0.2", "0.4", "sees all"])
    axes[1].set_xlabel("how far each fish sees (fraction of arena radius)")
    axes[1].set_title("Simulated share by visual range", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_simulations.png")


def make_figures(res, out, text=None):
    from experiments.e83_light_dose import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e83: pre-registered predictions", out)
    dose(res, out)
    sim_check(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e83_light_dose_raw.json").read_text()), Path("results/e83_figures"))
