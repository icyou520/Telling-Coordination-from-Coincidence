"""Figures for e80 (switching interaction off: tetras in light and dark, and real fish with virtual fish).

    uv run python -m experiments.e80_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def light(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained", sharey=True)
    rng = np.random.default_rng(0)
    for ax, twin, title in zip(axes, ("S_member", "S_iso"), ("member twin (own past)", "twin from fish recorded alone")):
        ticks = []
        for i, (lt, n) in enumerate([("Bright", 2), ("Dark", 2), ("Bright", 3), ("Dark", 3)]):
            rs = [r[twin] for r in res["light"] if r["kind"] == "real" and r["light"] == lt and r["n"] == n]
            sy = [r[twin] for r in res["light"] if r["kind"] == "synthetic" and r["light"] == lt and r["n"] == n]
            c = BLUE if lt == "Bright" else NAVY
            ax.plot(i + rng.uniform(-0.12, 0.12, len(rs)), rs, "o", color=c, ms=6, alpha=0.8)
            ax.plot(i + 0.3 + rng.uniform(-0.05, 0.05, len(sy)), sy, "x", color=GREY, ms=5)
            s = res["summary"][f"{lt}{n}"]["member" if twin == "S_member" else "iso"]
            ax.errorbar(i - 0.3, s["S"], yerr=[[s["S"] - s["ci"][0]], [s["ci"][1] - s["S"]]], fmt="s",
                        color=ORANGE, capsize=3, ms=6)
            ticks.append(f"{'pairs' if n == 2 else 'triads'}\n{lt.lower()}")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(4), ticks)
        ax.set_title(f"Interaction share, {title}", color=NAVY, loc="left", fontsize=10)
    axes[0].set_ylabel("S per recording (orange: pooled [90%]; grey x: synthetic groups of single fish)")
    _save(fig, out, "02_light_dark.png")


def vr(res, out):
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    rng = np.random.default_rng(0)
    for i, c in enumerate(("2R", "C1", "C2", "C3")):
        rs = [r for r in res["vr"] if r["condition"] == c]
        ax.plot(i + rng.uniform(-0.1, 0.1, len(rs)), [r["S_member"] for r in rs], "o", color=BLUE, ms=5, alpha=0.7)
        ax.plot(i + 0.25 + rng.uniform(-0.04, 0.04, len(rs)), [r["shift"] for r in rs], "x", color=GREY, ms=5)
        s = res["summary"][c]["member"]
        ax.errorbar(i - 0.25, s["S"], yerr=[[s["S"] - s["ci"][0]], [s["ci"][1] - s["S"]]], fmt="s", color=ORANGE,
                    capsize=3, ms=6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(4), ["two real\nfish", "C1: virtual\nresponds", "C2: faster clone\nresponds",
                             "C3: same clone\nignores the fish"])
    ax.set_ylabel("S_member (orange: pooled [90%]; grey x: shifted)")
    ax.set_title("Real fish with a virtual fish", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_virtual_reality.png")


def against_gap(res, out):
    fig, ax = plt.subplots(figsize=(5.6, 5), layout="constrained")
    for rs, c, lab in ((res["sim"], GREY, "simulations"), ([r for r in res["light"] if r["kind"] == "real"], BLUE,
                                                            "tetras, light and dark"), (res["vr"], ORANGE, "VR")):
        ax.plot([r["gap_member"] for r in rs], [r["S_member"] for r in rs], "o", color=c, ms=4, alpha=0.7, label=lab)
    lim = max(0.05, max(max(r["S_member"], r["gap_member"]) for r in res["sim"] + res["light"] + res["vr"]))
    ax.plot([-0.01, lim], [-0.01, lim], color="black", lw=0.6)
    ax.set_xlabel("memory gap (no twin fitted)")
    ax.set_ylabel("measured S_member")
    ax.legend(fontsize=8)
    _save(fig, out, "04_against_gap.png")


def make_figures(res, out, text=None):
    from experiments.e80_intervention import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e80: pre-registered predictions", out)
    light(res, out)
    vr(res, out)
    against_gap(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e80_intervention_raw.json").read_text()), Path("results/e80_figures"))
