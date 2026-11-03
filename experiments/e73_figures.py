"""Figures for e73 (ALM replication on Economo et al. 2018, silencing power).

    uv run python -m experiments.e73_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.e41_figures import BLUE, GREY, NAVY, RED, _save, plt, summary_text
from experiments.e71_figures import scorecard


def replication(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax = axes[0]
    a = res["replication"]
    ax.plot([r["S_pred"] for r in a], [r["S"] for r in a], "o", color=BLUE)
    lim = max(0.02, max(max(r["S"], r["S_pred"]) for r in a) * 1.1)
    ax.plot([0, lim], [0, lim], "--", color=GREY, lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("memory-gap prediction")
    ax.set_ylabel("measured share (cross-fitted)")
    ax.set_title("Economo et al. 2018 ALM sessions", color=NAVY, loc="left", fontsize=10)
    ax = axes[1]
    for cond, c in (("hidden", RED), ("coupled", BLUE)):
        rows = [v for v in res["power"].values() if v["cond"] == cond]
        ax.plot([v["silenced_trials"] for v in rows], [v["power"] for v in rows], "o-", color=c,
                label="weak input (ALM-sized share)" if cond == "hidden" else "coupled, no input (false positives)")
    ax.axhline(0.8, color=GREY, ls="--", lw=0.8)
    ax.axhline(0.05, color=GREY, ls=":", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("silenced trials per session")
    ax.set_ylabel("power, 21 sessions")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("How many silenced trials a decisive test needs", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "02_replication_power.png")


def make_figures(res, out, text=None):
    from experiments.e73_alm_replication import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e73: pre-registered predictions", out)
    replication(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e73_alm_replication_raw.json").read_text()), Path("results/e73_figures"))
