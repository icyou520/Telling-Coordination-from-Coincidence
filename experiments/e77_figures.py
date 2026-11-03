"""Figures for e77 (why the quadratic memory gap runs above the fitted fish twins).

    uv run python -m experiments.e77_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, RED, _save, plt, summary_text
from experiments.e71_figures import scorecard


def decomposition(res, out):
    from experiments.e77_fish_gap import published
    real = [r for r in res["real"] if r["kind"] == "real"]
    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    x = np.arange(len(real))
    for k, (key, label, c) in enumerate((("memory_interleaved", "memory twin, interleaved (e72)", ORANGE),
                                         ("full_interleaved", "full twin, interleaved", BLUE),
                                         ("full_chrono", "full twin, 70/30 split", NAVY),
                                         ("memory_chrono", "memory twin, 70/30 split", RED))):
        ax.plot(x + (k - 1.5) * 0.12, [r[key] for r in real], "o", color=c, ms=5, label=label)
    ax.plot(x, [published(r) for r in real], "_", color="black", ms=16, mew=2, label="published S_proj (e33, e38)")
    ax.set_xticks(x, [f"{'A' if r['set'] == 'adult' else 'Z'}{r['n']}\n{r['rec']}" for r in real], fontsize=7)
    ax.set_ylabel("interaction share")
    ax.set_title("Real schools: the same share under each twin and split (A = adult fish, Z = zebrafish)",
                 color=NAVY, loc="left", fontsize=10)
    ax.legend(fontsize=7.5, ncol=3, loc="lower left")
    _save(fig, out, "02_decomposition.png")


def simulations(res, out):
    sch = [r for r in res["sim"] if r["rule"] != "loners"]
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    t = np.array([r["true_interleaved"] for r in sch])
    for key, label, c in (("memory_interleaved", "memory twin", ORANGE), ("full_interleaved", "fitted full twin", BLUE),
                          ("proj_true", "e33 pipeline, true twin", GREY)):
        ax.plot(t, [r[key] for r in sch], "o", color=c, ms=5, label=label, alpha=0.8)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("true twin share (phi^2, interleaved)")
    ax.set_ylabel("share")
    ax.set_title("Simulated schools with a known twin", color=NAVY, loc="left", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    _save(fig, out, "03_simulations.png")


def make_figures(res, out, text=None):
    from experiments.e77_fish_gap import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e77: pre-registered predictions", out)
    decomposition(res, out)
    simulations(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e77_fish_gap_raw.json").read_text()), Path("results/e77_figures"))
