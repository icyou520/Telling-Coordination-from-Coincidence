"""Figures for e53 (influence map replication in zebrafish and pigeons), from the raw results.

    uv run python -m experiments.e53_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text
from experiments.e52_figures import CH_LABEL

DATA = (("zebrafish", "zebrafish (60-100 fish)"), ("pigeons", "pigeons (10 birds)"))


def _sel(res, data, kind):
    from experiments.e53_influence_replication import _sel as s
    return s(res, data, kind)


def _label(r):
    return r["group"].strip("()").replace("'", "").replace(", ", "-")


def scorecard(res, out):
    from experiments.e53_influence_replication import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e53: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=GREEN if ok else RED, transform=ax.transAxes))
        ax.text(0.03, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center", weight="bold",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.075, y + 0.015, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=10)
        ax.text(0.075, y - 0.025, detail[:170], va="center", transform=ax.transAxes, fontsize=7, color=GREY)
    _save(fig, out, "01_scorecard.png")


def checks(res, out):
    runs = res["runs"]
    labels = [f"{r['data'][0].upper()}{_label(r)}{'v' if r['kind'] == 'virtual' else ''}" for r in runs]
    x = np.arange(len(runs))
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), layout="constrained", sharex=True)
    a = [r["h"]["1"]["planted"]["after"] for r in runs]
    axes[0].bar(x - 0.2, [r["h"]["1"]["planted"]["before"]["G"] for r in runs], 0.4, color=GREY, label="before planting")
    axes[0].bar(x + 0.2, [v["G"] for v in a], 0.4, color=BLUE, label="with a planted 5% signal")
    axes[0].errorbar(x + 0.2, [v["G"] for v in a], yerr=[[v["G"] - v["lo"] for v in a], [v["hi"] - v["G"] for v in a]],
                     fmt="none", ecolor="black", lw=0.8)
    axes[0].set_title("Calibration: does the test see a planted signal?", color=NAVY, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, [r["h"]["1"]["null"]["G"] for r in runs], "o", color=ORANGE)
    axes[1].axhspan(-0.01, 0.01, color=GREEN, alpha=0.12)
    axes[1].set_title("Null control: influence features of another group (should be ~0)", color=NAVY, loc="left",
                      fontsize=10)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("G for the turn, 1 s")
    axes[1].set_xticks(x, labels, fontsize=7, rotation=60)
    _save(fig, out, "02_test_checks.png")


def comparison(res, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), layout="constrained", width_ratios=(3, 1.2))
    for col, (data, title) in enumerate(DATA):
        real, virt = _sel(res, data, "real"), _sel(res, data, "virtual")
        x = np.arange(len(real))
        for row, sec in enumerate(("1", "3")):
            ax = axes[row, col]
            for off, rows, k, c, lab in ((-0.3, real, "conf", GREY, "real: hand-made configuration"),
                                         (-0.1, real, "infl", ORANGE, "real: learned influence map"),
                                         (0.1, virt, "conf", "#c9ced8", "virtual: configuration"),
                                         (0.3, virt, "infl", BLUE, "virtual: influence map")):
                g = [r["h"][sec][k] for r in rows]
                ax.bar(x + off, [v["G"] for v in g], 0.2, color=c, label=lab)
                ax.errorbar(x + off, [v["G"] for v in g], yerr=[[v["G"] - v["lo"] for v in g],
                                                                [v["hi"] - v["G"] for v in g]],
                            fmt="none", ecolor="black", lw=0.6)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x, [_label(r) for r in real], fontsize=8)
            ax.set_title(f"{title}, {sec} s", color=NAVY, loc="left", fontsize=10)
        axes[0, col].set_ylabel("G for the group's turn") if col == 0 else None
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, out, "03_comparison.png")


def _norm_mean(rows, c):
    return np.mean([np.array(r["maps"][c]) / (np.abs(np.array(r["maps"][c])).max() or 1) for r in rows], 0)


def maps(res, out):
    from experiments.e53_influence_replication import GRID
    chans = ("sin", "turn", "speed_sin")
    combos = [(d, k) for d, _ in DATA for k in ("real", "virtual")]
    fig, axes = plt.subplots(len(combos), 3, figsize=(12.5, 15), layout="constrained")
    ext = [GRID[0], GRID[-1], GRID[0], GRID[-1]]
    for row, (data, kind) in enumerate(combos):
        rows = _sel(res, data, kind)
        dens = np.mean([np.array(r["density"]) for r in rows], 0)
        for col, c in enumerate(chans):
            ax = axes[row, col]
            im = ax.imshow(_norm_mean(rows, c), origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
            if dens.max() > 0:
                ax.contour(GRID, GRID, dens, levels=np.quantile(dens[dens > 0], [0.5, 0.9]), colors="black",
                           linewidths=0.6, alpha=0.5)
            ax.annotate("", xy=(2.2, -2.2), xytext=(1.2, -2.2), arrowprops=dict(arrowstyle="->", color=NAVY))
            ax.set_title(f"{data} {kind}: {CH_LABEL[c]}", color=NAVY, loc="left", fontsize=8.5)
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])
    fig.colorbar(im, ax=axes, shrink=0.4, label="push on the group's next turn (red = left), each group scaled to max 1")
    fig.suptitle("Influence maps in two more species (group moving to the right)", color=NAVY)
    _save(fig, out, "04_influence_maps.png")


def per_group(res, out):
    from experiments.e53_influence_replication import GRID
    rows = _sel(res, "zebrafish", "real") + _sel(res, "pigeons", "real")
    fig, axes = plt.subplots(2, 6, figsize=(16, 6), layout="constrained")
    ext = [GRID[0], GRID[-1], GRID[0], GRID[-1]]
    for ax, r in zip(axes.ravel(), rows):
        m = np.array(r["maps"]["sin"])
        ax.imshow(m / (np.abs(m).max() or 1), origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{r['data']} {_label(r)}", fontsize=8, color=NAVY)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("Heading map (sin φ) for every real group: red front = front individuals steer", color=NAVY)
    _save(fig, out, "05_per_group_maps.png")


def channels(res, out):
    from experiments.e53_influence_replication import GROUPS
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), layout="constrained")
    x = np.arange(len(GROUPS))
    for ax, (data, title) in zip(axes, DATA):
        for off, kind, c in ((-0.2, "real", ORANGE), (0.2, "virtual", GREY)):
            vals = np.array([[r["h"]["1"]["only_" + g]["G"] for g in GROUPS] for r in _sel(res, data, kind)])
            ax.bar(x + off, vals.mean(0), 0.4, color=c, label=f"{kind} (mean)")
            for v in vals:
                ax.plot(x + off, v, ".", color="black", ms=4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, list(GROUPS), fontsize=9)
        ax.set_title(f"{title}: which part of an individual's state carries the turn?", color=NAVY, loc="left",
                     fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("G for the turn, 1 s, one group alone")
    _save(fig, out, "06_channels.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, checks, comparison, maps, per_group, channels):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e53_influence_replication import analyze
    d = Path("results")
    data = json.loads((d / "e53_influence_replication_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e53_figures", txt)
