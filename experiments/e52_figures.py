"""Figures for e52 (a learned influence map for the school's turn), from the raw results.

    uv run python -m experiments.e52_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

CH_LABEL = {"sin": "heading left of the school (sin φ)", "turn": "already turning left (ω)",
            "place": "just being there", "cos": "aligned with the school (cos φ)",
            "speed": "faster than the school", "speed_sin": "faster and heading left"}


def _sel(res, n, kind):
    from experiments.e52_influence_map import _sel as s
    return s(res, n, kind)


def scorecard(res, out):
    from experiments.e52_influence_map import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(12, 0.8 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e52: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.04), 0.07, 0.08, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.035, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.09, y + 0.02, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.09, y - 0.045, detail, va="center", transform=ax.transAxes, fontsize=8, color=GREY)
    _save(fig, out, "01_scorecard.png")


def checks(res, out):
    groups = _sel(res, 8, "real") + _sel(res, 8, "virtual")
    labels = [f"{'R' if r['kind'] == 'real' else 'V'}{r['rec']}" for r in groups]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
    a = [r["h"]["10"]["planted"]["after"] for r in groups]
    axes[0].bar(x - 0.2, [r["h"]["10"]["planted"]["before"]["G"] for r in groups], 0.4, color=GREY,
                label="before planting")
    axes[0].bar(x + 0.2, [v["G"] for v in a], 0.4, color=BLUE, label="with a planted 5% signal")
    axes[0].errorbar(x + 0.2, [v["G"] for v in a], yerr=[[v["G"] - v["lo"] for v in a], [v["hi"] - v["G"] for v in a]],
                     fmt="none", ecolor="black", lw=1)
    axes[0].set_title("Calibration: does the test see a planted signal?", color=NAVY, loc="left", fontsize=10)
    axes[0].set_ylabel("G for the school's turn, 1 s")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, [r["h"]["10"]["null"]["G"] for r in groups], "o", color=ORANGE)
    axes[1].axhspan(-0.01, 0.01, color=GREEN, alpha=0.12)
    axes[1].set_title("Null control: influence features of another recording (should be ~0)", color=NAVY,
                      loc="left", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, labels, fontsize=8)
    _save(fig, out, "02_test_checks.png")


def comparison(res, out):
    real, virt, big = _sel(res, 8, "real"), _sel(res, 8, "virtual"), _sel(res, 39, "real")
    grp = real + virt + big
    labels = ([f"R{r['rec']}" for r in real] + [f"V{r['rec']}" for r in virt] + [f"39#{r['rec']}" for r in big])
    x = np.arange(len(grp))
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.5), layout="constrained", sharex=True)
    for ax, h in zip(axes, ("10", "30")):
        for off, k, c, lab in ((-0.27, "conf", GREY, "hand-made configuration (e51)"),
                               (0.0, "infl", ORANGE, "learned influence map"),
                               (0.27, "beyond_conf", BLUE, "influence map beyond the configuration")):
            g = [r["h"][h][k] for r in grp]
            ax.bar(x + off, [v["G"] for v in g], 0.27, color=c, label=lab)
            ax.errorbar(x + off, [v["G"] for v in g], yerr=[[v["G"] - v["lo"] for v in g], [v["hi"] - v["G"] for v in g]],
                        fmt="none", ecolor="black", lw=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(f"G for the school's turn, {int(h) // 10} s")
        ax.axvspan(5.5, 11.5, color=GREY, alpha=0.08)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[0].set_title("How much does each description of the school's interior add to forecasting its turn? "
                      "(shaded = virtual groups)", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "03_comparison.png")


def _norm_mean(rows, c):
    ms = [np.array(r["maps"][c]) for r in rows]
    return np.mean([m / (np.abs(m).max() or 1) for m in ms], 0)


def maps(res, out):
    from experiments.e52_influence_map import GRID
    real, virt = _sel(res, 8, "real"), _sel(res, 8, "virtual")
    chans = ("sin", "turn", "place", "speed_sin")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), layout="constrained")
    ext = [GRID[0], GRID[-1], GRID[0], GRID[-1]]
    dens = np.mean([np.array(r["density"]) for r in real], 0)
    for row, (rows, name) in enumerate(((real, "real schools"), (virt, "virtual groups"))):
        for col, c in enumerate(chans):
            ax = axes[row, col]
            m = _norm_mean(rows, c)
            im = ax.imshow(m, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.contour(GRID, GRID, dens, levels=np.quantile(dens[dens > 0], [0.5, 0.9]), colors="black",
                       linewidths=0.6, alpha=0.5)
            ax.annotate("", xy=(2.2, -2.2), xytext=(1.2, -2.2), arrowprops=dict(arrowstyle="->", color=NAVY))
            ax.text(1.2, -2.0, "swim", color=NAVY, fontsize=7)
            ax.set_title(f"{name}: {CH_LABEL[c]}", color=NAVY, loc="left", fontsize=9)
            ax.set_xlabel("ahead of centre (spreads)")
            ax.set_ylabel("left of centre (spreads)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="push on the school's next turn (red = left), each school scaled to max 1")
    fig.suptitle("Where in the school, and doing what, does a fish predict the school's next turn?", color=NAVY)
    _save(fig, out, "04_influence_maps.png")


def per_school(res, out):
    from experiments.e52_influence_map import GRID
    rows = _sel(res, 8, "real") + _sel(res, 39, "real")
    fig, axes = plt.subplots(2, len(rows), figsize=(2.6 * len(rows), 5.6), layout="constrained")
    ext = [GRID[0], GRID[-1], GRID[0], GRID[-1]]
    for j, r in enumerate(rows):
        for i, c in enumerate(("sin", "turn")):
            m = np.array(r["maps"][c])
            axes[i, j].imshow(m / (np.abs(m).max() or 1), origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"{r['N']} fish #{r['rec']}", fontsize=9, color=NAVY)
        axes[0, 0].set_ylabel("heading left (sin φ)")
        axes[1, 0].set_ylabel("turning left (ω)")
    fig.suptitle("Each real school separately (swimming to the right; red = predicts a left turn)", color=NAVY)
    _save(fig, out, "05_per_school_maps.png")


def channels(res, out):
    from experiments.e52_influence_map import GROUPS
    real, virt = _sel(res, 8, "real"), _sel(res, 8, "virtual")
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    x = np.arange(len(GROUPS))
    for off, rows, c, lab in ((-0.2, real, ORANGE, "real"), (0.2, virt, GREY, "virtual")):
        vals = np.array([[r["h"]["10"]["only_" + g]["G"] for g in GROUPS] for r in rows])
        ax.bar(x + off, vals.mean(0), 0.4, color=c, label=f"{lab} (mean of 6)")
        for v in vals:
            ax.plot(x + off, v, ".", color="black", ms=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f"{g}\n({', '.join(ch)})" for g, ch in GROUPS.items()], fontsize=9)
    ax.set_ylabel("G for the school's turn, 1 s, one group alone")
    ax.set_title("Which part of a fish's state carries the information?", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "06_channels.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, checks, comparison, maps, per_school, channels):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e52_influence_map import analyze
    d = Path("results")
    data = json.loads((d / "e52_influence_map_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e52_figures", txt)
