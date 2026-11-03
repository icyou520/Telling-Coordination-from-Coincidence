"""Figures for e71 (ALM holdout and silencing with the memory gap).

    uv run python -m experiments.e71_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def scorecard(preds, title, out):
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def holdout(res, out):
    a = res["holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax = axes[0]
    ax.plot([r["S_pred"] for r in a], [r["S"] for r in a], "o", color=BLUE, label="sessions")
    lim = max(0.02, max(max(r["S"], r["S_pred"]) for r in a) * 1.1)
    ax.plot([0, lim], [0, lim], "--", color=GREY, lw=0.8)
    ax.set_xlabel("memory-gap prediction")
    ax.set_ylabel("measured share (cross-fitted)")
    ax.set_title("Unseen ALM sessions: prediction against measurement", color=NAVY, loc="left", fontsize=10)
    ax = axes[1]
    cols = [("S", "measured", BLUE), ("drift", "beyond drift", ORANGE), ("virtual", "virtual", GREY)]
    for k, (key, label, c) in enumerate(cols):
        v = [r[key] for r in a]
        ax.plot(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v, "o", color=c, ms=5)
        ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color="black", lw=1.5)
    ax.set_xticks(range(3), [c[1] for c in cols])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("share")
    ax.set_title("Measured, beyond session drift, and virtual groups", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "02_holdout.png")


def silencing(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    conds = ["independent", "coupled", "hidden", "weak"]
    for k, c in enumerate(conds):
        d = [r["did"] for r in res["sim"] if r["cond"] == c]
        ax.plot(k + np.random.default_rng(k).uniform(-0.15, 0.15, len(d)), d, "o",
                color=RED if c in ("hidden", "weak") else BLUE, ms=4)
    ax.set_xticks(range(4), ["independent", "coupled", "hidden\ninput", "weak\ninput"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("difference-in-differences")
    ax.set_title("Simulated networks: silencing an input", color=NAVY, loc="left", fontsize=10)
    ax = axes[1]
    b = res["silencing"]
    order = np.argsort([r["did"] for r in b])
    ax.barh(range(len(b)), [b[i]["did"] for i in order], color=[RED if b[i]["did"] < 0 else BLUE for i in order])
    ax.set_yticks(range(len(b)), [b[i]["session"] for i in order], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("difference-in-differences (negative = share falls when the midbrain is silenced)")
    ax.set_title("ALM sessions", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_silencing.png")


def make_figures(res, out, text=None):
    from experiments.e71_alm_holdout import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e71: pre-registered predictions", out)
    holdout(res, out)
    silencing(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e71_alm_holdout_raw.json").read_text()), Path("results/e71_figures"))
