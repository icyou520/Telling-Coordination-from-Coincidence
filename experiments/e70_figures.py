"""Figures for e70 (exploratory memory-gap screen).

    uv run python -m experiments.e70_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

COLORS = {"fish": BLUE, "pigeons": GREEN, "ALM": ORANGE, "markets": RED, "brain areas": NAVY}


def screen(rows, out):
    brain = Path("results/e67_memory_gap_raw.json")
    areas = json.loads(brain.read_text())["areas"] if brain.exists() else []
    groups = {d: [r["S_pred"] for r in rows if r["domain"] == d] for d in dict.fromkeys(r["domain"] for r in rows)}
    virt = {d: [r["virtual"] for r in rows if r["domain"] == d] for d in groups}
    if areas:
        groups["brain areas"] = [a["S_pred"] for a in areas]
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    for k, (d, v) in enumerate(groups.items()):
        ax.plot(k + np.random.default_rng(k).uniform(-0.18, 0.18, len(v)), v, "o", color=COLORS.get(d, GREY),
                ms=4, alpha=0.75)
        if d in virt:
            ax.plot(np.full(len(virt[d]), k + 0.3), virt[d], "x", color=GREY, ms=4)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_yticks([0, 0.01, 0.03, 0.1, 0.3, 0.6], ["0", "0.01", "0.03", "0.1", "0.3", "0.6"])
    ax.set_ylim(-0.002, 1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("predicted interaction share (memory gap)")
    ax.set_title("Screening every dataset with the memory-gap equation (x = virtual baseline)", color=NAVY, loc="left")
    _save(fig, out, "01_screen.png")


def make_figures(rows, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    screen(rows, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e70_screen_raw.json").read_text()), Path("results/e70_figures"))
