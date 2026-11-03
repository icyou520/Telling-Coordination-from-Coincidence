"""Figures for e62 (the interaction share of brain areas, repaired twin), from the raw results.

    uv run python -m experiments.e62_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, RED, _save, plt, summary_text
from experiments.e61_figures import nested, real_areas

COND_LABEL = {"independent": "independent", "coupled": "strongly\ncoupled", "weak": "weakly\ncoupled",
              "hidden": "hidden drive", "recorded": "drive recorded", "coupled_rec": "coupled +\ndrive recorded"}


def scorecard(res, out):
    from experiments.e62_neural_share_repaired import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e62: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.025), 0.06, 0.05, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.012, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.02, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def simulations(res, out):
    from experiments.e62_neural_share_repaired import CONDITIONS
    fig, ax = plt.subplots(figsize=(11, 4.4), layout="constrained")
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
    raw = json.loads(Path("results/e62_neural_share_repaired_raw.json").read_text())
    make_figures(raw, Path("results/e62_figures"))
