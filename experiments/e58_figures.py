"""Figures for e58 (is openness carried by members' momentary innovations?), from the raw results.

    uv run python -m experiments.e58_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

MODELS = (("all", "all features", GREY), ("innov", "innovations", ORANGE), ("ownturn", "raw own turning", BLUE))


def _sel(res, system):
    from experiments.e58_innovation import _sel as s
    return s(res, system)


def scorecard(res, out):
    from experiments.e58_innovation import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e58: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def _bars(ax, rs, labels, keys):
    x = np.arange(len(rs))
    w = 0.8 / len(keys)
    for j, (k, lab, c) in enumerate(keys):
        ax.bar(x + (j - (len(keys) - 1) / 2) * w, [r["turn"][k]["G"] for r in rs], w, color=c, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, fontsize=7, rotation=60)
    ax.legend(frameon=False, fontsize=7)


def what_carries(res, out):
    fl, fish = _sel(res, "flock"), _sel(res, "fish")
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout="constrained")
    _bars(axes[0], fl, [f"{r['i']} m{r['params']['m']}" for r in fl], MODELS)
    axes[0].set_title("Flocks: what carries the group's next turn?", color=NAVY, loc="left", fontsize=10)
    _bars(axes[1], fish, [r["group"] for r in fish], MODELS)
    axes[1].set_title("Real fish: what carries the school's next turn (1 s)?", color=NAVY, loc="left", fontsize=10)
    for ax in axes:
        ax.set_ylabel("closure gap G")
    _save(fig, out, "02_what_carries.png")


def independence(res, out):
    fl, fish = _sel(res, "flock"), _sel(res, "fish")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained", width_ratios=(1, 1.4, 1.4))
    rng = np.random.default_rng(0)
    for r in fl:
        iota = np.array(r["iota"])
        m = r["params"]["m"]
        x = r["i"] + rng.uniform(-0.2, 0.2, len(iota))
        axes[0].plot(x[m:], iota[m:], ".", color=GREY, ms=3)
        axes[0].plot(x[:m], iota[:m], "o", color=ORANGE, ms=4)
    axes[0].set_xlabel("flock")
    axes[0].set_ylabel("independence (share of turning not explained locally)")
    axes[0].set_title("Do informed agents (orange) stand out?", color=NAVY, loc="left", fontsize=10)
    keys = (("indep", "most independent members", ORANGE), ("conform", "least independent members", BLUE))
    _bars(axes[1], fl, [str(r["i"]) for r in fl], keys)
    axes[1].set_title("Flocks: independent vs conforming members", color=NAVY, loc="left", fontsize=10)
    _bars(axes[2], fish, [r["group"] for r in fish], keys)
    axes[2].set_title("Real fish: independent vs conforming members", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_independence.png")


def checks(res, out):
    fl = _sel(res, "flock")
    fish = _sel(res, "fish")
    x = np.arange(len(fl))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), layout="constrained")
    a = [r["planted"]["after"] for r in fl]
    axes[0].bar(x - 0.2, [r["planted"]["before"]["G"] for r in fl], 0.4, color=GREY, label="before planting")
    axes[0].bar(x + 0.2, [v["G"] for v in a], 0.4, color=BLUE, label="with a planted 5% signal")
    axes[0].set_title("Calibration (flocks): does the test see a planted signal?", color=NAVY, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    runs = fl + fish
    axes[1].plot(np.arange(len(runs)), [r["turn"]["null"]["G"] for r in runs], "o", color=ORANGE)
    axes[1].axhspan(-0.01, 0.01, color=GREEN, alpha=0.12)
    axes[1].set_title("Null control: innovations of another group (should be ~0)", color=NAVY, loc="left", fontsize=10)
    axes[1].set_xticks(np.arange(len(runs)), [str(r.get("i", r.get("group"))) for r in runs], fontsize=6, rotation=60)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("G")
    _save(fig, out, "04_test_checks.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, what_carries, independence, checks):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e58_innovation import analyze
    d = Path("results")
    data = json.loads((d / "e58_innovation_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e58_figures", txt)
