"""Figures for e48 (closed group-level layer in real fish), from the raw results.

    uv run python -m experiments.e48_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

PURPLE = "#7b5ea7"


def _pairs(res):
    from experiments.e48_fish_layers import _pairs as p
    return p(res)


def scorecard(res, out):
    from experiments.e48_fish_layers import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e48: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.035), 0.07, 0.07, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.035, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.09, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.09, y - 0.04, detail, va="center", transform=ax.transAxes, fontsize=8, color=GREY)
    _save(fig, out, "01_scorecard.png")


def closure(res, out):
    real, virt = _pairs(res)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), layout="constrained", sharey=True)
    x = np.arange(len(real))
    for j, (ax, lab) in enumerate(zip(axes, ("1 s ahead", "3 s ahead"))):
        ax.bar(x - 0.3, [r["G"]["group"][j] for r in real], 0.2, color=ORANGE, label="group block, real school")
        ax.bar(x - 0.1, [v["G"]["group"][j] for v in virt], 0.2, color=GREY, label="group block, virtual group")
        ax.bar(x + 0.1, [r["G"]["fish0"][j] for r in real], 0.2, color=BLUE, label="one fish, real school")
        rnd = [np.median([r["G"][f"random{k}"][j] for k in range(10)]) for r in real]
        ax.bar(x + 0.3, rnd, 0.2, color=PURPLE, label="random 5-dim blocks (median), real")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(0.1, color=RED, ls="--", lw=1)
        ax.set_xticks(x, [f"8 fish #{r['rec']}" for r in real], fontsize=8)
        ax.set_title(f"Closure gap, {lab} (0 = the block predicts itself)", color=NAVY, loc="left")
    axes[0].set_ylabel("closure gap G")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out, "02_closure.png")


def interaction(res, out):
    real, virt = _pairs(res)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained", sharey=True)
    x = np.arange(len(real))
    for j, (ax, lab) in enumerate(zip(axes, ("1 s ahead", "3 s ahead"))):
        ax.bar(x - 0.2, [r["Sh"]["group"][j] for r in real], 0.4, color=ORANGE, label="real school")
        ax.bar(x + 0.2, [v["Sh"]["group"][j] for v in virt], 0.4, color=GREY, label="virtual group")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [f"8 fish #{r['rec']}" for r in real], fontsize=8)
        ax.set_title(f"Group block: interaction share, {lab}", color=NAVY, loc="left")
    axes[0].set_ylabel("interaction share S_h")
    axes[0].legend(frameon=False)
    _save(fig, out, "03_interaction_share.png")


def spectra(res, out):
    real, virt = _pairs(res)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.4), layout="constrained", sharey=True)
    for ax, r, v in zip(axes.flat, real, virt):
        k = np.arange(1, len(r["S_test"]) + 1)
        ax.plot(k, np.array(r["S_test"])[::-1], "o-", ms=3, color=ORANGE, label="real school")
        ax.plot(k, np.array(v["S_test"])[::-1], "o-", ms=3, color=GREY, label="virtual group")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(5.5, color=GREEN, ls="--", lw=1)
        ax.set_title(f"8 fish #{r['rec']} (bottom-5 vs group block: {r['overlap_group']:.2f})",
                     color=NAVY, fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out S at 1 s")
    for ax in axes[1]:
        ax.set_xlabel("collective variable, most blind first (40 = 8 fish x 5)")
    _save(fig, out, "04_spectra.png")


def blind_weights(res, out):
    real, _ = _pairs(res)
    from experiments.e48_fish_layers import NAMES, group_block
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.4), layout="constrained")
    for ax, r in zip(axes.flat, real):
        w = np.array(r["w"])[:, -5:]
        t = group_block(8)
        u, _, vt = np.linalg.svd(w.T @ t)
        wr = w @ (u @ vt)
        img = np.abs(wr).sum(1).reshape(5, 8)
        ax.imshow(img / img.max(), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(5), NAMES)
        ax.set_xticks(range(8), [str(i + 1) for i in range(8)])
        ax.set_title(f"8 fish #{r['rec']}: weight of the 5 most blind", color=NAVY, fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("fish")
    _save(fig, out, "05_blind_weights.png")


def layer_map(res, out):
    real, virt = _pairs(res)
    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    for rows, mk, kind in ((real, "o", "real"), (virt, "s", "virtual")):
        for name, col, lab in (("group", ORANGE, "group block"), ("fish0", BLUE, "one fish"),
                               ("bottom5", GREEN, "5 most blind"), ("top5", RED, "5 most interaction-made")):
            gx = [max(r["G"][name]) for r in rows]
            sy = [max(r["Sh"][name]) for r in rows]
            ax.scatter(gx, sy, marker=mk, color=col, s=60 if kind == "real" else 30,
                       alpha=1.0 if kind == "real" else 0.5, label=f"{lab} ({kind})")
    ax.axvline(0.1, color=GREY, ls="--", lw=1)
    ax.axhline(0.05, color=GREY, ls="--", lw=1)
    ax.set_xlabel("largest closure gap (1 s, 3 s)")
    ax.set_ylabel("largest interaction share (1 s, 3 s)")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    ax.set_title("Which blocks of fish variables are layers? (circles real, squares virtual)",
                 color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "06_layer_map.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, closure, interaction, spectra, blind_weights, layer_map):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e48_fish_layers import analyze
    d = Path("results")
    data = json.loads((d / "e48_fish_layers_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e48_figures", txt)
