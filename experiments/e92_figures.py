"""Figures for e92 (Onsager vortex clusters, point-vortex simulation).

    uv run python -m experiments.e92_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def share_vs_energy(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    groups = {"non-interacting": GREY, "dipoles": BLUE, "random": NAVY, "clusters": ORANGE}
    for g, col in groups.items():
        rows = [r for k, r in res.items() if k.startswith(g)]
        ax.scatter([r["energy"] for r in rows], [r["S"] for r in rows], color=col, label=g, s=28)
        ax.scatter([r["energy"] for r in rows], [r["S_jitter"] for r in rows], color=col, marker="x", s=22)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("energy per vortex (Onsager: high = same-sign clusters)")
    ax.set_ylabel("interaction share S (dots clean, crosses with imaging jitter)")
    ax.set_title("20 vortices, 2,000 frames", color=NAVY, loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    seeds = sorted({k.rsplit(" ", 1)[1] for k in res})
    names = ["small random", "small clusters 0.15"]
    x = np.arange(len(seeds))
    for j, (n, col) in enumerate(zip(names, (NAVY, ORANGE))):
        rows = [res[f"{n}, seed {s}"] for s in seeds]
        ax.bar(x + (j - 0.5) * 0.38, [r["S"] for r in rows], width=0.36, color=col, label=f"{n} (clean)")
        ax.scatter(x + (j - 0.5) * 0.38, [r["S_jitter"] for r in rows], color="black", marker="x", zorder=3,
                   label="with jitter" if j else None)
    ax.set_xticks(x, [f"seed {s}" for s in seeds])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("8 vortices, 500 frames", color=NAVY, loc="left", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, out, "02_share_vs_energy.png")


def make_figures(res, out, text=None):
    from experiments.e92_vortices import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e92: pre-registered predictions", out)
    share_vs_energy(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e92_vortices_raw.json").read_text()), Path("results/e92_figures"))
