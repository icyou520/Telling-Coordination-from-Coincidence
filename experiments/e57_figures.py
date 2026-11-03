"""Figures for e57 (oscillators: does measured looseness predict and locate openness?), from the raw results.

    uv run python -m experiments.e57_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def scorecard(res, out):
    from experiments.e57_oscillator_looseness import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e57: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def _err(ax, x, g, **kw):
    ax.errorbar(x, [v["G"] for v in g], yerr=[[max(0, v["G"] - v["lo"]) for v in g],
                                               [max(0, v["hi"] - v["G"]) for v in g]], **kw)


def main_result(res, out):
    rs = res["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    _err(axes[0], [r["max_lam"] for r in rs], [r["r"]["loosest"] for r in rs], fmt="o", color=BLUE, ecolor=GREY,
         lw=0.8, label="3 loosest members")
    axes[0].plot([r["max_lam"] for r in rs], [r["r"]["tightest"]["G"] for r in rs], "x", color=ORANGE,
                 label="3 tightest members")
    axes[0].set_xlabel("looseness of the loosest member")
    axes[0].set_title("Synchrony r: does a loose member open the group?", color=NAVY, loc="left", fontsize=10)
    _err(axes[1], [r["sum_lam"] for r in rs], [r["drift"]["loosest"] for r in rs], fmt="o", color=ORANGE,
         ecolor=GREY, lw=0.8, label="3 loosest members")
    axes[1].set_xlabel("total looseness (sum over the 30 members)")
    axes[1].set_title("Phase drift: does it open with the amount of looseness?", color=NAVY, loc="left", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.axhspan(-0.02, 0.02, color=GREEN, alpha=0.1)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("closure gap G (1 time unit)")
    fig.suptitle("30 random oscillator groups: measured looseness against openness", color=NAVY)
    _save(fig, out, "02_main_result.png")


def checks(res, out):
    runs = sorted(res["runs"], key=lambda r: r["max_lam"])
    x = np.arange(len(runs))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), layout="constrained", sharex=True)
    a = [r["planted"]["after"] for r in runs]
    axes[0].bar(x - 0.2, [r["planted"]["before"]["G"] for r in runs], 0.4, color=GREY, label="before planting")
    axes[0].bar(x + 0.2, [v["G"] for v in a], 0.4, color=BLUE, label="with a planted 5% signal")
    axes[0].set_title("Calibration: does the test see a planted signal?", color=NAVY, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x - 0.1, [r["r"]["null"]["G"] for r in runs], "s", color=BLUE, label="r")
    axes[1].plot(x + 0.1, [r["drift"]["null"]["G"] for r in runs], "o", color=ORANGE, label="drift")
    axes[1].axhspan(-0.01, 0.01, color=GREEN, alpha=0.12)
    axes[1].set_title("Null control: loosest members of another run (should be ~0)", color=NAVY, loc="left",
                      fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("G")
    axes[1].set_xticks(x, [f"{r['i']}\n{r['max_lam']:.2f}" for r in runs], fontsize=6)
    axes[1].set_xlabel("group (and its max looseness), sorted by looseness")
    _save(fig, out, "03_test_checks.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, main_result, checks):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e57_oscillator_looseness import analyze
    d = Path("results")
    data = json.loads((d / "e57_oscillator_looseness_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e57_figures", txt)
