"""Figures for e49 (relative headings across group size; the school's shape as a layer).

    uv run python -m experiments.e49_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

PURPLE = "#7b5ea7"


def _sel(res, n, kind):
    from experiments.e49_school_layer import _sel as s
    return s(res, n, kind)


def scorecard(res, out):
    from experiments.e49_school_layer import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e49: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.035), 0.07, 0.07, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.035, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.09, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.09, y - 0.04, detail, va="center", transform=ax.transAxes, fontsize=8, color=GREY)
    _save(fig, out, "01_scorecard.png")


def spectra(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), layout="constrained")
    for r in _sel(res, 8, "real"):
        asc = np.array(r["spectrum"]["S_test"])[::-1]
        axes[0].plot(np.arange(1, 17), asc, "o-", ms=3, color=ORANGE, alpha=0.8)
    for v in _sel(res, 8, "virtual"):
        asc = np.array(v["spectrum"]["S_test"])[::-1]
        axes[0].plot(np.arange(1, 17), asc, "o-", ms=3, color=GREY, alpha=0.6)
    axes[0].plot([], [], "o-", color=ORANGE, label="6 real schools")
    axes[0].plot([], [], "o-", color=GREY, label="6 virtual groups")
    axes[0].axvline(2.5, color=GREEN, ls="--", lw=1.2, label="predicted step: 2 x (8 - 1) = 14 above")
    axes[0].set_xlabel("heading variable, most blind first (16 = 8 fish x 2)")
    axes[0].set_title("8 fish", color=NAVY, loc="left")
    for r, col in zip(_sel(res, 39, "real"), (ORANGE, RED)):
        asc = np.array(r["spectrum"]["S_test"])[::-1]
        axes[1].plot(np.arange(1, len(asc) + 1), asc, "o-", ms=3, color=col,
                     label=f"39 fish #{r['rec']} (top block found: {r['spectrum']['top']})")
    axes[1].axvline(2.5, color=GREEN, ls="--", lw=1.2, label="predicted step: 2 x 38 = 76 above")
    axes[1].set_xlabel("heading variable, most blind first (78 = 39 fish x 2)")
    axes[1].set_title("39 fish", color=NAVY, loc="left")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("held-out S at 1 s")
        ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "02_heading_spectra.png")


def shape_share(res, out):
    real, virt, big = _sel(res, 8, "real"), _sel(res, 8, "virtual"), _sel(res, 39, "real")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained", sharey=True)
    for ax, idx, name in zip(axes, (5, 6, 7), ("polarization", "spread", "milling")):
        x = np.arange(len(real))
        for j, (h, off, alpha) in enumerate((("10", -0.3, 1.0), ("30", -0.1, 0.6))):
            ax.bar(x + off, [r["Sh"][h][idx] for r in real], 0.2, color=ORANGE, alpha=alpha,
                   label=f"real, {int(h) // 10} s")
            ax.bar(x + off + 0.4, [v["Sh"][h][idx] for v in virt], 0.2, color=GREY, alpha=alpha,
                   label=f"virtual, {int(h) // 10} s")
        xb = len(real) + 0.5 + np.arange(len(big))
        ax.bar(xb, [r["Sh"]["10"][idx] for r in big], 0.3, color=RED, label="39 fish, 1 s")
        ax.set_xticks(list(x) + list(xb), [f"8 #{r['rec']}" for r in real] + [f"39 #{r['rec']}" for r in big],
                      fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(0.3, color=GREEN, ls="--", lw=1)
        ax.set_title(f"{name}: interaction share", color=NAVY, loc="left")
    axes[0].set_ylabel("S_h (full vs twin forecast)")
    axes[0].legend(frameon=False, fontsize=7.5, ncols=2)
    _save(fig, out, "03_shape_interaction_share.png")


def closure(res, out):
    real, virt = _sel(res, 8, "real"), _sel(res, 8, "virtual")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), layout="constrained")
    x = np.arange(len(real))
    for j, (h, off) in enumerate((("1 s", -0.3), ("3 s", 0.1))):
        axes[0].bar(x + off, [r["G_school"][j] for r in real], 0.2, color=ORANGE,
                    alpha=1.0 if j == 0 else 0.6, label=f"real, {h}")
        axes[0].bar(x + off + 0.2, [v["G_school"][j] for v in virt], 0.2, color=GREY,
                    alpha=1.0 if j == 0 else 0.6, label=f"virtual, {h}")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].axhline(0.05, color=GREEN, ls="--", lw=1)
    axes[0].set_xticks(x, [f"8 fish #{r['rec']}" for r in real], fontsize=8)
    axes[0].set_ylabel("closure gap G")
    axes[0].set_title("School state: do the fish's own tendencies add anything?", color=NAVY, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    for r, v, xi in zip(real, virt, x):
        axes[1].scatter(np.full(8, xi - 0.12), r["G_fish"], color=ORANGE, s=18)
        axes[1].scatter(np.full(8, xi + 0.12), v["G_fish"], color=GREY, s=18)
    axes[1].scatter([], [], color=ORANGE, label="real (each fish)")
    axes[1].scatter([], [], color=GREY, label="virtual (each fish)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x, [f"8 fish #{r['rec']}" for r in real], fontsize=8)
    axes[1].set_ylabel("closure gap G, 1 s")
    axes[1].set_title("A single fish: does it need the others?", color=NAVY, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, out, "04_closure.png")


def forecasts(res, out):
    real = _sel(res, 8, "real")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
    x = np.arange(len(real))
    for ax, idx, name, unit in ((axes[0], 5, "polarization", ""), (axes[1], 6, "spread", " (cm)")):
        for off, key, col, lab in ((-0.25, "obs", "black", "observed"), (0.0, "full", ORANGE, "model with interactions"),
                                   (0.25, "twin", GREY, "twin (no interactions)")):
            ax.bar(x + off, [r["obs_vs_forecast_30"][key][idx] for r in real], 0.25, color=col, label=lab)
        ax.set_xticks(x, [f"8 fish #{r['rec']}" for r in real], fontsize=8)
        ax.set_title(f"{name} 3 s ahead{unit}", color=NAVY, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out, "05_shape_forecasts.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, spectra, shape_share, closure, forecasts):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e49_school_layer import analyze
    d = Path("results")
    data = json.loads((d / "e49_school_layer_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e49_figures", txt)
