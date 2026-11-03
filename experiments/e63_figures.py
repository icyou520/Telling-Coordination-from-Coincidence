"""Figures for e63 (is a brain area's population state a closed level?), from the raw results.

    uv run python -m experiments.e63_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

HORIZONS = (5, 10, 20)


def scorecard(res, out):
    from experiments.e63_area_closure import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e63: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def checks(res, out):
    r = res["real"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    gain = [x["planted"]["after"]["G"] - x["planted"]["before"]["G"] for x in r]
    axes[0].hist(gain, bins=30, color=BLUE)
    axes[0].axvline(0.025, color=RED, ls="--", lw=1)
    axes[0].set_xlabel("rise in G from a planted 5% signal")
    axes[0].set_title("The test finds a planted signal", color=NAVY, loc="left")
    axes[1].hist([x["state_5_null"]["G"] for x in r], bins=30, color=GREY)
    axes[1].axvspan(-0.01, 0.01, color=GREEN, alpha=0.15)
    axes[1].set_xlabel("G from null features (another trial)")
    axes[1].set_title("and nothing from features without information", color=NAVY, loc="left")
    _save(fig, out, "02_checks.png")


def horizons(res, out):
    r = res["real"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, tgt in zip(axes, ("state", "rate")):
        for k, (extra, c) in enumerate((("internal", BLUE), ("other", ORANGE), ("null", GREY))):
            data = [[x[f"{tgt}_{h}_{extra}"]["G"] for x in r if f"{tgt}_{h}_{extra}" in x] for h in HORIZONS]
            pos = np.arange(len(HORIZONS)) + (k - 1) * 0.25
            b = ax.boxplot(data, positions=pos, widths=0.2, patch_artist=True, showfliers=False)
            for p in b["boxes"]:
                p.set_facecolor(c)
                p.set_alpha(0.6)
            ax.plot([], [], "s", color=c, label=extra)
        ax.set_xticks(range(len(HORIZONS)), [f"{h * 10} ms" for h in HORIZONS])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"target: {'top 3 components' if tgt == 'state' else 'total activity'}", color=NAVY, loc="left")
        ax.set_ylabel("closure gap G")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("How open is a brain area's summary state, and to what?", color=NAVY)
    _save(fig, out, "03_horizons.png")


def by_area(res, out):
    r = res["real"]
    names = sorted({x["area"] for x in r},
                   key=lambda a: np.median([x["state_5_internal"]["G"] for x in r if x["area"] == a]))
    fig, ax = plt.subplots(figsize=(max(10, 0.38 * len(names)), 4.6), layout="constrained")
    for k, a in enumerate(names):
        xs = [x for x in r if x["area"] == a]
        ax.plot(np.full(len(xs), k - 0.15), [x["state_5_internal"]["G"] for x in xs], "o", color=BLUE, ms=3,
                label="internal" if k == 0 else None)
        ax.plot(np.full(len(xs), k + 0.15), [x.get("state_5_other", {}).get("G", np.nan) for x in xs], "o",
                color=ORANGE, ms=3, label="other areas" if k == 0 else None)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("G (top 3 components, 50 ms)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Which areas are open, and to what", color=NAVY, loc="left")
    _save(fig, out, "04_by_area.png")


def link(res, out):
    e62 = res.get("e62", {})
    r = [x for x in res["real"] if f"{x['session']}|{x['area']}" in e62]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].plot([e62[f"{x['session']}|{x['area']}"] for x in r], [x["rate_5_internal"]["G"] for x in r], "o",
                 color=BLUE, ms=4)
    axes[0].set_xlabel("interaction share S_proj beyond all recorded causes (e62)")
    axes[0].set_ylabel("G_internal (total activity, 50 ms)")
    axes[0].set_title("Is interaction what opens the level?", color=NAVY, loc="left")
    for p, c in (("pre", GREY), ("response", ORANGE), ("late", BLUE)):
        v = [x["state_5_internal"]["periods"][p]["G"] for x in res["real"] if p in x["state_5_internal"]["periods"]]
        axes[1].hist(v, bins=30, histtype="step", color=c, lw=1.5, label=p)
    axes[1].set_xlabel("G_internal (top 3 components, 50 ms)")
    axes[1].set_title("Before, during and after the stimulus response", color=NAVY, loc="left")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.5) if ax is axes[0] else ax.axvline(0, color="black", lw=0.5)
    _save(fig, out, "05_link_and_periods.png")


def make_figures(res, out, text=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(res, out)
    checks(res, out)
    horizons(res, out)
    by_area(res, out)
    link(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    raw = json.loads(Path("results/e63_area_closure_raw.json").read_text())
    make_figures(raw, Path("results/e63_figures"))
