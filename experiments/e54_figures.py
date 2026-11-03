"""Figures for e54 (stable variables close, neutral directions stay open?), from the raw results.

    uv run python -m experiments.e54_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

SYSTEMS = (("flock", "flock (30 agents)"), ("kuramoto", "oscillators (30, Kuramoto)"))


def _runs(res, system):
    return sorted([r for r in res["runs"] if r["system"] == system], key=lambda r: (r["field"], r["seed"]))


def _err(ax, x, g, **kw):
    ax.errorbar(x, [v["G"] for v in g], yerr=[[max(0, v["G"] - v["lo"]) for v in g],
                                               [max(0, v["hi"] - v["G"]) for v in g]], **kw)


def scorecard(res, out):
    from experiments.e54_neutral_directions import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e54: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def main_result(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for ax, (system, title) in zip(axes, SYSTEMS):
        rs = _runs(res, system)
        f = np.array([r["field"] for r in rs])
        pos = {v: i for i, v in enumerate(sorted(set(f)))}
        x = np.array([pos[v] for v in f]) + np.array([(r["seed"] - 0.5) * 0.15 for r in rs])
        _err(ax, x - 0.05, [r["turn"]["arrangement"] for r in rs], fmt="o", color=ORANGE, ecolor=ORANGE, lw=1,
             label="the group's direction (turn)")
        _err(ax, x + 0.05, [r["stable"]["arrangement"] for r in rs], fmt="s", color=BLUE, ecolor=BLUE, lw=1,
             label="its stable variables")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhspan(-0.02, 0.02, color=GREEN, alpha=0.1)
        ax.set_xticks(range(len(pos)), [f"{v:g}" for v in pos])
        ax.set_xlabel("strength of the external field pinning the direction")
        ax.set_title(title, color=NAVY, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("closure gap G (what the inside adds, 1 time unit)")
    fig.suptitle("Pinning the neutral direction: does the group's turn become closed?", color=NAVY)
    _save(fig, out, "02_main_result.png")


def checks(res, out):
    runs = sorted(res["runs"], key=lambda r: (r["system"], r["field"], r["seed"]))
    labels = [f"{r['system'][0].upper()} f{r['field']:g} s{r['seed']}" for r in runs]
    x = np.arange(len(runs))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), layout="constrained", sharex=True)
    a = [r["planted"]["after"] for r in runs]
    axes[0].bar(x - 0.2, [r["planted"]["before"]["G"] for r in runs], 0.4, color=GREY, label="before planting")
    axes[0].bar(x + 0.2, [v["G"] for v in a], 0.4, color=BLUE, label="with a planted 5% signal")
    _err(axes[0], x + 0.2, a, fmt="none", ecolor="black", lw=0.8)
    axes[0].set_title("Calibration: does the test see a planted signal?", color=NAVY, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x - 0.1, [r["turn"]["null"]["G"] for r in runs], "o", color=ORANGE, label="turn")
    axes[1].plot(x + 0.1, [r["stable"]["null"]["G"] for r in runs], "s", color=BLUE, label="stable variables")
    axes[1].axhspan(-0.01, 0.01, color=GREEN, alpha=0.12)
    axes[1].set_title("Null control: the arrangement of another run (should be ~0)", color=NAVY, loc="left",
                      fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("G")
    axes[1].set_xticks(x, labels, fontsize=7, rotation=60)
    _save(fig, out, "03_test_checks.png")


def pinning(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for system, title in SYSTEMS:
        rs = _runs(res, system)
        f = [r["field"] for r in rs]
        axes[0].plot(f, [r["psi_sd"] for r in rs], "o-" if system == "flock" else "s--", label=title)
        axes[1].plot(f, [r["turn_sd"] for r in rs], "o-" if system == "flock" else "s--", label=title)
    axes[0].set_yscale("log")
    axes[0].set_title("How far the direction wanders (sd, radians)", color=NAVY, loc="left", fontsize=10)
    axes[1].set_title("Size of the turn in one time unit (sd, radians)", color=NAVY, loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("field strength")
        ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "04_pinning.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, main_result, checks, pinning):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e54_neutral_directions import analyze
    d = Path("results")
    data = json.loads((d / "e54_neutral_directions_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e54_figures", txt)
