"""Figures for e51 (a trusted closure test), from the raw results.

    uv run python -m experiments.e51_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text


def _sel(res, n, kind):
    from experiments.e51_trusted_closure import _sel as s
    return s(res, n, kind)


def scorecard(res, out):
    from experiments.e51_trusted_closure import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(12, 0.8 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e51: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
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
    """The two checks on the test itself: planted signal (calibration) and null features."""
    groups = _sel(res, 8, "real") + _sel(res, 8, "virtual")
    labels = [f"{'R' if r['kind'] == 'real' else 'V'}{r['rec']}" for r in groups]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), layout="constrained")
    for ax, t, name in ((axes[0], "turn_planted", "heading change"), (axes[1], "spread_planted", "spread change")):
        b = [r["h"]["10"][t]["before"]["G"] for r in groups]
        a = [r["h"]["10"][t]["after"] for r in groups]
        ax.bar(x - 0.2, b, 0.4, color=GREY, label="before planting")
        ax.bar(x + 0.2, [v["G"] for v in a], 0.4, color=BLUE, label="with a planted 5% signal")
        ax.errorbar(x + 0.2, [v["G"] for v in a], yerr=[[v["G"] - v["lo"] for v in a], [v["hi"] - v["G"] for v in a]],
                    fmt="none", ecolor="black", lw=1)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(f"Calibration: does the test see a planted signal? ({name})", color=NAVY, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("closure gap G, 1 s")
    w = 0.27
    for k, (t, c) in enumerate((("turn", ORANGE), ("motion", BLUE), ("shape", GREEN))):
        axes[2].plot(x + (k - 1) * w, [r["h"]["10"][t]["null"]["G"] for r in groups], "o", color=c, label=t)
    axes[2].axhspan(-0.01, 0.01, color=GREEN, alpha=0.12)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(x, labels, fontsize=8)
    axes[2].set_title("Null control: configuration from another recording (should be ~0)", color=NAVY, loc="left",
                      fontsize=10)
    axes[2].legend(frameon=False, fontsize=8)
    _save(fig, out, "02_test_checks.png")


def versus_e50(res, out):
    groups = _sel(res, 8, "real") + _sel(res, 8, "virtual")
    labels = [f"{'R' if r['kind'] == 'real' else 'V'}{r['rec']}" for r in groups]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
    for ax, t in zip(axes, ("turn", "shape")):
        ax.bar(x - 0.2, [r["h"]["10"][t]["ols_conf"] for r in groups], 0.4, color=GREY, label="e50-style least squares")
        ax.bar(x + 0.2, [r["h"]["10"][t]["conf"]["G"] for r in groups], 0.4, color=BLUE, label="e51 cross-validated ridge")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(f"Same features, two forecasters: {t}", color=NAVY, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("closure gap G, 1 s")
    _save(fig, out, "03_versus_e50.png")


def main_result(res, out):
    real, virt, big = _sel(res, 8, "real"), _sel(res, 8, "virtual"), _sel(res, 39, "real")
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), layout="constrained", sharey="col")
    for row, h in enumerate(("10", "30")):
        for col, t in enumerate(("turn", "motion", "shape", "all")):
            ax = axes[row, col]
            for k, (grp, c, lab) in enumerate(((real, ORANGE, "real 8"), (virt, GREY, "virtual 8"),
                                                (big, RED, "real 39"))):
                xs = np.arange(len(grp)) + 7 * k * (k < 2) + (14 if k == 2 else 0)
                g = [r["h"][h][t]["conf"] for r in grp]
                ax.bar(xs, [v["G"] for v in g], 0.8, color=c, label=lab)
                ax.errorbar(xs, [v["G"] for v in g], yerr=[[v["G"] - v["lo"] for v in g], [v["hi"] - v["G"] for v in g]],
                            fmt="none", ecolor="black", lw=0.8)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks([2.5, 9.5, 14.5], ["real 8", "virtual 8", "39"], fontsize=8)
            ax.set_title(f"{t}, {int(h) // 10} s", color=NAVY, loc="left", fontsize=10)
        axes[row, 0].set_ylabel("G (+configuration), 95% interval")
    fig.suptitle("Does the school's internal configuration help forecast the school? (0 = school state suffices)",
                 color=NAVY)
    _save(fig, out, "04_configuration_closure.png")


def leadership_share(res, out):
    real, big = _sel(res, 8, "real"), _sel(res, 39, "real")
    grp = real + big
    labels = [f"8 #{r['rec']}" for r in real] + [f"39 #{r['rec']}" for r in big]
    x = np.arange(len(grp))
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    ax.bar(x - 0.2, [r["h"]["10"]["turn"]["conf"]["G"] for r in grp], 0.4, color=ORANGE, label="all configuration")
    ax.bar(x + 0.2, [r["h"]["10"]["turn"]["conf_no_lead"]["G"] for r in grp], 0.4, color=GREY,
           label="without the leadership features")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("G for the school's heading change, 1 s")
    ax.set_title("How much of the configuration's information about turning is leadership?", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "05_leadership_share.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, checks, versus_e50, main_result, leadership_share):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e51_trusted_closure import analyze
    d = Path("results")
    data = json.loads((d / "e51_trusted_closure_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e51_figures", txt)
