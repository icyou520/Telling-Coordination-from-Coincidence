"""Figures for e67 (one equation for the interaction share: the memory gap).

    uv run python -m experiments.e67_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def scorecard(res, out):
    from experiments.e67_memory_gap import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e67: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def predicted_vs_measured(res, out):
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    styles = {"factor": (BLUE, "o", "common input"), "meanfield": (ORANGE, "s", "all-to-all coupling"),
              "local": (RED, "D", "local coupling"), "garch": (GREEN, "^", "simulated markets"),
              "spiking": (NAVY, "v", "spiking networks")}
    for kind, (c, m, lab) in styles.items():
        rs = [r for r in res["sim"] if r["kind"] == kind]
        ax.plot([r["S_pred"] for r in rs], [r["S"] for r in rs], m, color=c, ms=5, alpha=0.8, label=lab)
    areas = res.get("areas", [])
    ax.plot([a["S_pred"] for a in areas], [a["S"] for a in areas], ".", color=GREY, ms=4, alpha=0.6,
            label=f"brain areas ({len(areas)})")
    if res.get("industries"):
        r = res["industries"]
        ax.plot([r["S_pred"]], [r["S"]], "*", color="black", ms=12, label="US industries")
    lim = [-0.02, max(0.7, max(r["S"] for r in res["sim"]) + 0.03)]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("predicted from the two memories (S_pred)")
    ax.set_ylabel("measured interaction share (S)")
    ax.set_title("One equation, every system", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "02_predicted_vs_measured.png")


def pooling_peak(res, out):
    from experiments.e67_memory_gap import N_GRID, R_GRID, RHO_C
    fac = [r for r in res["sim"] if r["kind"] == "factor"]
    fig, axes = plt.subplots(1, len(RHO_C), figsize=(12, 4.2), layout="constrained", sharey=True)
    for ax, rho in zip(axes, RHO_C):
        for n, c in zip(N_GRID, (BLUE, ORANGE, RED)):
            rs = sorted([r for r in fac if r["n"] == n and r["rho_c"] == rho], key=lambda r: r["p"])
            if not rs:
                continue
            ax.plot([r["p"] for r in rs], [r["S"] for r in rs], "o", color=c, label=f"N = {n} measured")
            ax.plot([r["p"] for r in rs], [r["S_pred"] for r in rs], "-", color=c, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("share of each member's variance that is the common signal")
        ax.set_title(f"Common input with memory {rho}", color=NAVY, loc="left")
    axes[0].set_ylabel("interaction share")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out, "03_pooling_peak.png")


def memories(res, out):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    pts = [(r["rho_m"], r["rho_g"], r["S"]) for r in res["sim"]] + [(a["rho_m"], a["rho_g"], a["S"])
                                                                   for a in res.get("areas", [])]
    if res.get("industries"):
        r = res["industries"]
        pts.append((r["rho_m"], r["rho_g"], r["S"]))
    m, g, s = map(np.array, zip(*pts))
    sc = ax.scatter(m, g, c=np.clip(s, 0, None), cmap="viridis", s=18)
    lim = [min(m.min(), g.min()) - 0.05, 1.0]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1)
    fig.colorbar(sc, ax=ax, label="measured S")
    ax.set_xlabel("member memory (lag-1 autocorrelation)")
    ax.set_ylabel("group memory (lag-1 autocorrelation)")
    ax.set_title("S is the gap between the two memories", color=NAVY, loc="left")
    _save(fig, out, "04_memories.png")


def make_figures(res, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(res, out)
    predicted_vs_measured(res, out)
    pooling_peak(res, out)
    memories(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    raw = json.loads(Path("results/e67_memory_gap_raw.json").read_text())
    make_figures(raw, Path("results/e67_figures"))
