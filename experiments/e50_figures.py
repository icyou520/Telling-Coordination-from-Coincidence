"""Figures for e50 (school configuration and leadership), from the raw results.

    uv run python -m experiments.e50_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def _sel(res, n, kind):
    from experiments.e50_school_configuration import _sel as s
    return s(res, n, kind)


def scorecard(res, out):
    from experiments.e50_school_configuration import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(11, 0.8 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e50: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.04), 0.07, 0.08, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.035, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.09, y + 0.02, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.09, y - 0.045, detail, va="center", transform=ax.transAxes, fontsize=8, color=GREY)
    _save(fig, out, "01_scorecard.png")


def closure(res, out):
    real, virt, big = _sel(res, 8, "real"), _sel(res, 8, "virtual"), _sel(res, 39, "real")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), layout="constrained", sharey=True)
    labels = [f"8 #{r['rec']}" for r in real] + [f"39 #{r['rec']}" for r in big]
    x = np.arange(len(labels))
    for ax, key, thr in ((axes[0], "motion", 0.02), (axes[1], "shape", 0.05)):
        for off, h, alpha in ((-0.2, "10", 1.0), (0.0, "30", 0.55)):
            ax.bar(x[:6] + off, [r["G"][h][key] for r in real], 0.2, color=ORANGE, alpha=alpha,
                   label=f"real, {int(h) // 10} s")
            ax.bar(x[6:] + off, [r["G"][h][key] for r in big], 0.2, color=RED, alpha=alpha,
                   label=f"39 fish, {int(h) // 10} s")
        ax.bar(x[:6] + 0.2, [v["G"]["10"][key] for v in virt], 0.2, color=GREY, label="virtual, 1 s")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(thr, color=GREEN, ls="--", lw=1)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(f"Does the internal configuration help forecast the school's {key}?",
                     color=NAVY, loc="left", fontsize=10)
    axes[0].set_ylabel("closure gap G (0 = school state suffices)")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out, "02_configuration_closure.png")


def per_variable(res, out):
    from experiments.e49_school_layer import Z_NAMES
    rows = _sel(res, 8, "real") + _sel(res, 39, "real") + _sel(res, 8, "virtual")
    mat = np.array([r["G"]["10"]["per_variable"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    lim = max(0.05, np.abs(mat).max())
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(Z_NAMES)), Z_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [f"{r['N']} fish {r['kind']} #{r['rec']}" for r in rows], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="1 - MSE(+configuration) / MSE(school state), 1 s")
    ax.set_title("Which school variables does the configuration help forecast?", color=NAVY, loc="left")
    _save(fig, out, "03_per_variable.png")


def leadership(res, out):
    real, virt, big = _sel(res, 8, "real"), _sel(res, 8, "virtual"), _sel(res, 39, "real")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
    labels = [f"8 #{r['rec']}" for r in real] + [f"39 #{r['rec']}" for r in big]
    x = np.arange(len(labels))
    axes[0].bar(x[:6] - 0.2, [r["lead"]["t"] for r in real], 0.4, color=ORANGE, label="real")
    axes[0].bar(x[:6] + 0.2, [v["lead"]["t"] for v in virt], 0.4, color=GREY, label="virtual")
    axes[0].bar(x[6:], [r["lead"]["t"] for r in big], 0.4, color=RED, label="39 fish")
    axes[0].axhline(3, color=GREEN, ls="--", lw=1)
    axes[0].axhline(-3, color=GREEN, ls="--", lw=1)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("t (Newey-West)")
    axes[0].set_title("Do fish at the front steer the school's turn over the next 1 s?", color=NAVY, loc="left",
                      fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x[:6] - 0.2, [r["speed_lead_t"] for r in real], 0.4, color=ORANGE, label="real")
    axes[1].bar(x[:6] + 0.2, [v["speed_lead_t"] for v in virt], 0.4, color=GREY, label="virtual")
    axes[1].bar(x[6:], [r["speed_lead_t"] for r in big], 0.4, color=RED, label="39 fish")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_ylabel("t of mean(relative speed x front position)")
    axes[1].set_title("Are the front fish the faster ones?", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "04_leadership.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, closure, per_variable, leadership):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e50_school_configuration import analyze
    d = Path("results")
    data = json.loads((d / "e50_school_configuration_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e50_figures", txt)
