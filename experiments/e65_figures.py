"""Figures for e65 (does ALM's coordination depend on its midbrain input? pre-cue window).

    uv run python -m experiments.e65_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def scorecard(res, out):
    from experiments.e65_input_silencing_precue import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e65: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def simulations(res, out):
    from experiments.e65_input_silencing_precue import SIM_CONDITIONS
    labels = {"independent": "independent", "coupled": "local coupling", "hidden": "hidden input\n(switched off)"}
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for k, c in enumerate(SIM_CONDITIONS):
        for r in [r for r in res["sim"] if r["cond"] == c]:
            ax.plot([k - 0.15, k + 0.15], [r["precue"]["control"]["S"], r["precue"]["silenced"]["S"]], "-o",
                    color=ORANGE if c == "hidden" else BLUE, ms=4, alpha=0.7)
    ax.set_xticks(range(len(SIM_CONDITIONS)), [labels[c] for c in SIM_CONDITIONS])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-0.03, 0.03, color=GREEN, alpha=0.1)
    ax.set_ylabel("interaction share before the cue\n(left: control, right: silenced)")
    ax.set_title("Simulated networks: the redesigned test", color=NAVY, loc="left")
    _save(fig, out, "02_simulations.png")


def sessions(res, out):
    from experiments.e65_input_silencing_precue import effects
    real = res["real"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    for ax, w in zip(axes, ("baseline", "precue")):
        for r in real:
            ax.plot([0, 1], [r[w]["control"]["S"], r[w]["silenced"]["S"]], "-o", color=BLUE, ms=3, alpha=0.6)
        ax.plot([0, 1], [np.median([r[w]["control"]["S"] for r in real]),
                         np.median([r[w]["silenced"]["S"] for r in real])], "-s", color=NAVY, lw=3, ms=7,
                label="median")
        v = [r[w][k]["S"] for r in real for k in ("silenced_virtual", "control_virtual")]
        ax.axhspan(np.quantile(v, 0.1), np.quantile(v, 0.9), color=GREY, alpha=0.15, label="virtual (10-90%)")
        ax.set_xticks([0, 1], ["control", "midbrain silenced"])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("before the light" if w == "baseline" else "light on, before the cue", color=NAVY, loc="left")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("ALM interaction share")
    dp, db = zip(*[effects(r) for r in real])
    axes[2].hist(np.array(dp) - np.array(db), bins=12, color=BLUE)
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_xlabel("change from silencing, minus change before the light")
    axes[2].set_title("Difference-in-differences per session", color=NAVY, loc="left")
    _save(fig, out, "03_sessions.png")


def make_figures(res, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(res, out)
    simulations(res, out)
    if res["real"]:
        sessions(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    raw = json.loads(Path("results/e65_input_silencing_precue_raw.json").read_text())
    make_figures(raw, Path("results/e65_figures"))
