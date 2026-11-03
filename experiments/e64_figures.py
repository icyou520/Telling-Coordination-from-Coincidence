"""Figures for e64 (does a brain area's interaction share survive switching off a major input?).

    uv run python -m experiments.e64_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def scorecard(res, out):
    from experiments.e64_input_silencing import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e64: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def simulations(res, out):
    from experiments.e64_input_silencing import SIM_CONDITIONS
    labels = {"independent": "independent", "coupled": "local coupling\n(cue input removed)",
              "hidden": "hidden input\n(switched off)"}
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for k, c in enumerate(SIM_CONDITIONS):
        rs = [r for r in res["sim"] if r["cond"] == c]
        for r in rs:
            ax.plot([k - 0.15, k + 0.15], [r["light"]["control"]["S"], r["light"]["silenced"]["S"]], "-o",
                    color=ORANGE if c == "hidden" else BLUE, ms=4, alpha=0.8)
    ax.set_xticks(range(len(SIM_CONDITIONS)), [labels[c] for c in SIM_CONDITIONS])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("interaction share during the light\n(left: control, right: silenced)")
    ax.set_title("Simulated networks: does the test see an input switched off?", color=NAVY, loc="left")
    _save(fig, out, "02_simulations.png")


def sessions(res, out):
    real = res["real"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, w in zip(axes, ("baseline", "light")):
        for r in real:
            ax.plot([0, 1], [r[w]["control"]["S"], r[w]["silenced"]["S"]], "-o", color=BLUE, ms=3, alpha=0.6)
        ax.plot([0, 1], [np.median([r[w]["control"]["S"] for r in real]),
                         np.median([r[w]["silenced"]["S"] for r in real])], "-s", color=NAVY, lw=3, ms=7,
                label="median")
        v = [r[w]["silenced_virtual"]["S"] for r in real] + [r[w]["control_virtual"]["S"] for r in real]
        ax.axhspan(np.quantile(v, 0.1), np.quantile(v, 0.9), color=GREY, alpha=0.15, label="virtual (10-90%)")
        ax.set_xticks([0, 1], ["control", "midbrain silenced"])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("before the light (delay)" if w == "baseline" else "during the light (around the go cue)",
                     color=NAVY, loc="left")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("ALM interaction share")
    _save(fig, out, "03_sessions.png")


def effect(res, out):
    from experiments.e64_input_silencing import effects
    real = res["real"]
    dl, db = zip(*[effects(r) for r in real])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    axes[0].hist(np.array(dl) - np.array(db), bins=15, color=BLUE)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("change in S from silencing (light) minus change before the light")
    axes[0].set_title("Does switching off the midbrain lower ALM's share?", color=NAVY, loc="left")
    axes[1].plot([r["rise"]["control"] for r in real], [r["rise"]["silenced"] for r in real], "o", color=ORANGE)
    lim = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]), max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
    axes[1].plot(lim, lim, color=GREY, ls="--", lw=1)
    axes[1].set_xlabel("ALM response to the go cue, control (spikes per bin)")
    axes[1].set_ylabel("silenced")
    axes[1].set_title("Did silencing work? (below the line = yes)", color=NAVY, loc="left")
    _save(fig, out, "04_effect.png")


def make_figures(res, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(res, out)
    simulations(res, out)
    if res["real"]:
        sessions(res, out)
        effect(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    raw = json.loads(Path("results/e64_input_silencing_raw.json").read_text())
    make_figures(raw, Path("results/e64_figures"))
