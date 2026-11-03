"""Figures for e56 (does openness follow measured looseness?), from the raw results.

    uv run python -m experiments.e56_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def _sel(res, system):
    from experiments.e56_measured_lock import _sel as s
    return s(res, system)


def scorecard(res, out):
    from experiments.e56_measured_lock import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e56: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def looseness_vs_openness(res, out):
    ks, fs = _sel(res, "kuramoto"), _sel(res, "flock")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    panels = ((axes[0], ks, "max_lam", "stable", "oscillators: loosest member vs G(r)", "max looseness"),
              (axes[1], ks, "n_loose", "turn", "oscillators: loose members vs G(phase drift)",
               "members with looseness > 0.5"),
              (axes[2], fs, "max_lam", "turn", "flocks: loosest member vs G(turn)", "max looseness"))
    for ax, rs, key, t, title, xl in panels:
        g = [r[t]["all"] for r in rs]
        ax.errorbar([r[key] for r in rs], [v["G"] for v in g],
                    yerr=[[max(0, v["G"] - v["lo"]) for v in g], [max(0, v["hi"] - v["G"]) for v in g]],
                    fmt="o", color=ORANGE if t == "turn" else BLUE, ecolor=GREY, lw=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(xl)
        ax.set_title(title, color=NAVY, loc="left", fontsize=10)
    axes[0].set_ylabel("closure gap G (1 time unit)")
    fig.suptitle("Random groups: does the group open up when members are measured to be loose?", color=NAVY)
    _save(fig, out, "02_looseness_vs_openness.png")


def loosest_vs_tightest(res, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained", width_ratios=(1, 1, 1.4))
    for ax, s, t, title in ((axes[0], "kuramoto", "stable", "oscillators, G(r)"),
                            (axes[1], "flock", "turn", "flocks, G(turn)"),
                            (axes[2], "fish", "turn", "real fish, G(turn), 1 s")):
        rs = _sel(res, s)
        if s != "fish":
            rs = [r for r in rs if r[t]["all"]["G"] >= 0.03 and r[t]["all"]["lo"] > 0]
        x = np.arange(len(rs))
        ax.bar(x - 0.2, [r[t]["loosest"]["G"] for r in rs], 0.4, color=ORANGE, label="loosest members only")
        ax.bar(x + 0.2, [r[t]["tightest"]["G"] for r in rs], 0.4, color=BLUE, label="tightest members only")
        ax.plot(x, [r[t]["all"]["G"] for r in rs], "_", color="black", ms=12, label="all members")
        ax.axhline(0, color="black", lw=0.8)
        labels = [r.get("group", str(r.get("i"))).replace("zebrafish ", "Z").replace("puy ", "P") for r in rs]
        ax.set_xticks(x, labels, fontsize=7, rotation=60 if s == "fish" else 0)
        ax.set_title(title + ("" if s == "fish" else " (open runs only)"), color=NAVY, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("G from the arrangement of ...")
    _save(fig, out, "03_loosest_vs_tightest.png")


def checks(res, out):
    runs = _sel(res, "kuramoto") + _sel(res, "flock")
    x = np.arange(len(runs))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), layout="constrained", sharex=True)
    a = [r["planted"]["after"] for r in runs]
    axes[0].bar(x - 0.2, [r["planted"]["before"]["G"] for r in runs], 0.4, color=GREY, label="before planting")
    axes[0].bar(x + 0.2, [v["G"] for v in a], 0.4, color=BLUE, label="with a planted 5% signal")
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
    axes[1].set_xticks(x, [f"{r['system'][0].upper()}{r['i']}" for r in runs], fontsize=7, rotation=60)
    _save(fig, out, "04_test_checks.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, looseness_vs_openness, loosest_vs_tightest, checks):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e56_measured_lock import analyze
    d = Path("results")
    data = json.loads((d / "e56_measured_lock_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e56_figures", txt)
