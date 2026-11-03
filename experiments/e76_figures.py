"""Figures for e76 (a leak in the cross-fitted projected share).

    uv run python -m experiments.e76_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, RED, _save, plt, summary_text
from experiments.e71_figures import scorecard


def against_gap(res, out):
    """Measured share against the memory gap, leaky and leak-free: simulations, fresh markets, brain areas."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    panels = (
        ("Simulated markets (mean, breadth)", [r[v] for r in res["sim"] for v in ("mean", "breadth")]),
        ("Fresh markets: Japan, Europe, Asia Pacific", [r[v] for r in res["regions"] for v in ("mean", "breadth")]),
        ("Brain areas (e67)", res["areas"]),
    )
    for ax, (title, cells) in zip(axes, panels):
        p = np.array([c["S_pred"] for c in cells])
        k = np.array([c["leaky"] for c in cells])
        s = np.array([c["S"] for c in cells])
        ax.plot(p, k, "o", color=ORANGE, ms=4, alpha=0.7, label="leaky (as published)")
        ax.plot(p, s, "o", color=BLUE, ms=4, alpha=0.7, label="leak-free")
        lo, hi = min(p.min(), k.min(), s.min()), max(p.max(), k.max(), s.max())
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("memory gap S_pred (no fitting)")
        ax.set_title(title, color=NAVY, loc="left", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("measured share (cross-fitted)")
    _save(fig, out, "02_against_gap.png")


def regions(res, out):
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    labels, x = [], 0
    for r in res["regions"]:
        for v in ("mean", "breadth"):
            c = r[v]
            ax.bar(x - 0.27, c["leaky"], 0.27, color=ORANGE)
            ax.bar(x, c["S"], 0.27, color=BLUE, yerr=[[c["S"] - c["lo"]], [c["hi"] - c["S"]]], capsize=3)
            ax.bar(x + 0.27, c["S_pred"], 0.27, color=RED)
            ax.plot([x + 0.45] * len(r["virtual"][v]), r["virtual"][v], "x", color=GREY, ms=4)
            labels.append(f"{r['region'].replace('_', ' ').replace('Asia Pacific ex Japan', 'Asia Pac.')}\n{v}")
            x += 1
    ax.set_xticks(range(x), labels, fontsize=7.5)
    ax.axhline(0, color="black", lw=0.8)
    from matplotlib.patches import Patch
    handles = [Patch(color=ORANGE, label="leaky"), Patch(color=BLUE, label="leak-free (90% interval)"),
               Patch(color=RED, label="memory gap"),
               plt.Line2D([], [], marker="x", ls="", color=GREY, label="virtual groups (leak-free)")]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left")
    ax.set_ylabel("interaction share")
    ax.set_title("Fresh markets, 1990-2026", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_regions.png")


def make_figures(res, out, text=None):
    from experiments.e76_leak_audit import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e76: pre-registered predictions", out)
    against_gap(res, out)
    regions(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e76_leak_audit_raw.json").read_text()), Path("results/e76_figures"))
