"""Figures for e60 (is a level open when there is any slow internal direction?), from the raw results.

    uv run python -m experiments.e60_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

FAMILIES = (("oscillators", BLUE, "o"), ("flocks", ORANGE, "s"), ("fish", GREEN, "^"))


def scorecard(res, out):
    from experiments.e60_slow_features import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e60: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def main_result(res, out):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), layout="constrained")
    for ax, (fam, c, mk) in zip(axes, FAMILIES):
        rs = [r for r in res["runs"] if r["family"] == fam]
        ax.plot([r["tau"] for r in rs], [r["all"]["G"] for r in rs], mk, color=c, ms=6)
        ax.set_xscale("log")
        ax.set_title(fam, color=NAVY, loc="left", fontsize=10)
        ax.set_xlabel("memory time of the slowest internal direction")
    for fam, c, mk in FAMILIES:
        rs = [r for r in res["runs"] if r["family"] == fam]
        axes[3].plot([r["tau"] / r["h_time"] for r in rs], [r["all"]["G"] for r in rs], mk, color=c, ms=6, label=fam)
    axes[3].set_xscale("log")
    axes[3].axvline(1, color=GREY, ls="--", lw=1)
    axes[3].set_xlabel("slowest memory time / forecast horizon")
    axes[3].set_title("all groups pooled", color=NAVY, loc="left", fontsize=10)
    axes[3].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.axhspan(-0.02, 0.02, color=GREEN, alpha=0.1)
    axes[0].set_ylabel("closure gap G (1 time unit)")
    fig.suptitle("Is a level open when some internal direction is slow?", color=NAVY)
    _save(fig, out, "02_main_result.png")


def slow_vs_fast(res, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    for ax, (fam, _, _) in zip(axes, FAMILIES):
        rs = sorted([r for r in res["runs"] if r["family"] == fam], key=lambda r: r["all"]["G"])
        x = np.arange(len(rs))
        ax.plot(x, [r["all"]["G"] for r in rs], "_", color="black", ms=10, label="all features")
        ax.bar(x - 0.2, [r["slow"]["G"] for r in rs], 0.4, color=ORANGE, label="3 slowest directions")
        ax.bar(x + 0.2, [r["fast"]["G"] for r in rs], 0.4, color=BLUE, label="3 fastest directions")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks([])
        ax.set_xlabel("groups, sorted by G")
        ax.set_title(f"{fam}: where is the information?", color=NAVY, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("closure gap G")
    _save(fig, out, "03_slow_vs_fast.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, main_result, slow_vs_fast):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e60_slow_features import analyze
    d = Path("results")
    data = json.loads((d / "e60_slow_features_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e60_figures", txt)
