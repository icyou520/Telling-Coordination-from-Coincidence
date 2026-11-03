"""Figures for e55 (is a level open exactly where some members are not locked to it?), from the raw results.

    uv run python -m experiments.e55_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

SYSTEMS = (("flock", "flock (30 agents): free = privately informed"),
           ("kuramoto", "oscillators (30): free = unlocked"))
LABEL = {"flock": {"turn": "the group's turn", "stable": "polarization and spread"},
         "kuramoto": {"turn": "the group's phase drift", "stable": "synchrony r"}}


def _runs(res, system):
    return sorted([r for r in res["runs"] if r["system"] == system], key=lambda r: (r["m"], r["seed"]))


def _err(ax, x, g, **kw):
    ax.errorbar(x, [v["G"] for v in g], yerr=[[max(0, v["G"] - v["lo"]) for v in g],
                                               [max(0, v["hi"] - v["G"]) for v in g]], **kw)


def _x(rs):
    pos = {v: i for i, v in enumerate(sorted({r["m"] for r in rs}))}
    return np.array([pos[r["m"]] + (r["seed"] - 0.5) * 0.15 for r in rs]), pos


def scorecard(res, out):
    from experiments.e55_free_members import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e55: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
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
        x, pos = _x(rs)
        for off, t, c, mk in ((-0.05, "turn", ORANGE, "o"), (0.05, "stable", BLUE, "s")):
            _err(ax, x + off, [r[t]["arrangement"] for r in rs], fmt=mk, color=c, ecolor=c, lw=1,
                 label=LABEL[system][t])
        ax.axhline(0, color="black", lw=0.8)
        ax.axhspan(-0.02, 0.02, color=GREEN, alpha=0.1)
        ax.set_xticks(range(len(pos)), [str(v) for v in pos])
        ax.set_xlabel("number of free members (of 30)")
        ax.set_title(title, color=NAVY, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("closure gap G (what the inside adds, 1 time unit)")
    fig.suptitle("Does a group open up as members come loose?", color=NAVY)
    _save(fig, out, "02_main_result.png")


def free_vs_locked(res, out):
    from experiments.e55_free_members import AFFECTED
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")
    for ax, (system, title) in zip(axes, SYSTEMS):
        rs = [r for r in _runs(res, system) if r["m"]]
        aff = AFFECTED[system]
        x, pos = _x(rs)
        for off, k, c, lab in ((-0.12, "arrangement", GREY, "all members"), (0.0, "free", ORANGE, "free members only"),
                               (0.12, "locked", BLUE, "locked members only")):
            ax.bar(x + off, [r[aff][k]["G"] for r in rs], 0.06, color=c, label=lab)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(pos)), [str(v) for v in pos])
        ax.set_xlabel("number of free members")
        ax.set_title(f"{title}: G for {LABEL[system][aff]}", color=NAVY, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("G from the arrangement of ...")
    _save(fig, out, "03_free_vs_locked.png")


def checks(res, out):
    runs = sorted(res["runs"], key=lambda r: (r["system"], r["m"], r["seed"]))
    labels = [f"{r['system'][0].upper()} m{r['m']} s{r['seed']}" for r in runs]
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
    _save(fig, out, "04_test_checks.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, main_result, free_vs_locked, checks):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e55_free_members import analyze
    d = Path("results")
    data = json.loads((d / "e55_free_members_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e55_figures", txt)
