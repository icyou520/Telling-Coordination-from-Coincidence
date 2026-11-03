"""Figures for e61 (the interaction share of brain areas), from the raw results.

    uv run python -m experiments.e61_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

COND_LABEL = {"independent": "independent", "coupled": "coupled", "hidden": "hidden drive",
              "recorded": "drive recorded", "coupled_rec": "coupled +\ndrive recorded"}
ENVS = (("E0", "stimulus", BLUE), ("E1", "+ task, behaviour", ORANGE), ("E2", "+ other areas", GREEN))


def scorecard(res, out):
    from experiments.e61_neural_share import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e61: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def simulations(res, out):
    from experiments.e61_neural_share import CONDITIONS
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    for k, c in enumerate(CONDITIONS):
        rs = [r for r in res["sim"] if r["cond"] == c]
        xs = k + np.linspace(-0.12, 0.12, len(rs))
        ax.errorbar(xs, [r["E0"]["S"] for r in rs],
                    yerr=[[max(0, r["E0"]["S"] - r["E0"]["lo"]) for r in rs],
                          [max(0, r["E0"]["hi"] - r["E0"]["S"]) for r in rs]],
                    fmt="o", color=BLUE, ms=5, label="network" if k == 0 else None)
        ax.plot(xs + 0.3, [r["virtual"]["S"] for r in rs], "x", color=GREY, label="virtual" if k == 0 else None)
    ax.set_xticks(range(len(CONDITIONS)), [COND_LABEL[c] for c in CONDITIONS])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-0.02, 0.02, color=GREEN, alpha=0.1)
    ax.set_ylabel("interaction share S_proj")
    ax.legend(frameon=False)
    ax.set_title("Simulated spiking networks with known answers", color=NAVY, loc="left")
    _save(fig, out, "02_simulations.png")


def real_areas(res, out):
    real = sorted(res["real"], key=lambda r: np.median([q["E0"]["S"] for q in res["real"] if q["area"] == r["area"]]))
    names = list(dict.fromkeys(r["area"] for r in real))
    fig, ax = plt.subplots(figsize=(max(10, 0.38 * len(names)), 5), layout="constrained")
    for off, (key, label, c) in zip((-0.2, 0.0, 0.2), ENVS):
        for k, a in enumerate(names):
            v = [r[key]["S"] for r in real if r["area"] == a]
            ax.plot(np.full(len(v), k + off), v, "o", color=c, ms=3, alpha=0.6, label=label if k == 0 else None)
    for k, a in enumerate(names):
        v = [r["virtual"]["S"] for r in real if r["area"] == a]
        ax.plot(np.full(len(v), k + 0.35), v, "x", color=GREY, ms=4, label="virtual" if k == 0 else None)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("interaction share S_proj (50 ms)")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.set_title("Mouse brain areas: activity shaped by coordination beyond recorded causes", color=NAVY, loc="left")
    _save(fig, out, "03_real_areas.png")


def nested(res, out):
    real = res["real"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    data = [[r[k]["S"] for r in real] for k, _, _ in ENVS] + [[r["virtual"]["S"] for r in real]]
    axes[0].boxplot(data, tick_labels=[lab for _, lab, _ in ENVS] + ["virtual"])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("S_proj")
    axes[0].set_title("Adding recorded common causes", color=NAVY, loc="left")
    axes[1].plot([r["n"] for r in real], [r["E2"]["S"] for r in real], "o", color=GREEN, ms=4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("neurons in the area")
    axes[1].set_ylabel("S_proj beyond all recorded causes (E2)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Does it depend on how many neurons are recorded?", color=NAVY, loc="left")
    _save(fig, out, "04_nested.png")


def make_figures(res, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(res, out)
    simulations(res, out)
    if res["real"]:
        real_areas(res, out)
        nested(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    raw = json.loads(Path("results/e61_neural_share_raw.json").read_text())
    make_figures(raw, Path("results/e61_figures"))
