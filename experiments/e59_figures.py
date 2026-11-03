"""Figures for e59 (is a level open when its internal state is slow?), from the raw results.

    uv run python -m experiments.e59_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

FAMILIES = (("oscillators", BLUE, "o"), ("flocks", ORANGE, "s"), ("fish", GREEN, "^"))


def scorecard(res, out):
    from experiments.e59_slaving import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e59: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
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
        ax.set_xlabel("internal memory time (time units)")
    for fam, c, mk in FAMILIES:
        rs = [r for r in res["runs"] if r["family"] == fam]
        axes[3].plot([r["tau"] / r["h_time"] for r in rs], [r["all"]["G"] for r in rs], mk, color=c, ms=6, label=fam)
    axes[3].set_xscale("log")
    axes[3].axvline(1, color=GREY, ls="--", lw=1)
    axes[3].set_xlabel("internal memory time / forecast horizon")
    axes[3].set_title("all groups pooled", color=NAVY, loc="left", fontsize=10)
    axes[3].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.axhspan(-0.02, 0.02, color=GREEN, alpha=0.1)
    axes[0].set_ylabel("closure gap G (1 time unit)")
    fig.suptitle("Is a level open when its internal state is slow?", color=NAVY)
    _save(fig, out, "02_main_result.png")


def memory_curves(res, out):
    lags = np.array(res["lags"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), layout="constrained", sharey=True)
    for ax, (fam, c, _) in zip(axes, FAMILIES):
        rs = [r for r in res["runs"] if r["family"] == fam]
        gmax = max(max(r["all"]["G"] for r in rs), 1e-6)
        for r in rs:
            ax.plot(lags * r["dt"], r["r2"], color=plt.cm.viridis(np.clip(r["all"]["G"] / gmax, 0, 1)), lw=1)
        ax.axhline(np.exp(-1), color=GREY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.0)
        ax.set_xlabel("lag (time units)")
        ax.set_title(f"{fam}: memory of the internal state (yellow = most open)", color=NAVY, loc="left", fontsize=9)
    axes[0].set_ylabel("R² of the internal state, predicted from itself")
    _save(fig, out, "03_memory_curves.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, main_result, memory_curves):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e59_slaving import analyze
    d = Path("results")
    data = json.loads((d / "e59_slaving_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e59_figures", txt)
