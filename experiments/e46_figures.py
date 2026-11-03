"""Figures for e46 (spectrum with estimated twins; real fish), from the raw results.

    uv run python -m experiments.e46_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

PURPLE = "#7b5ea7"


def _unit_sum(w):
    w = np.asarray(w, float)
    return w / w.sum()


def _signed(w):
    w = np.asarray(w, float)
    return w * np.sign(w.sum() if abs(w.sum()) > 1e-9 else w[np.argmax(np.abs(w))]) / np.abs(w).max()


def scorecard(res, out):
    from experiments.e46_spectrum_estimated_twin import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e46: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.07, 0.06, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.035, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.09, y + 0.012, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.09, y - 0.033, detail, va="center", transform=ax.transAxes, fontsize=8.5, color=GREY)
    _save(fig, out, "01_scorecard.png")


def linear(res, out):
    r = res["linear"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained",
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x = np.arange(10)
    ax.bar(x - 0.27, _unit_sum(r["pi"]), 0.27, color=GREEN, label="truth (left null vector pi)")
    ax.bar(x, _unit_sum(np.array(r["w"])[:, -1]), 0.27, color=BLUE, label="data-only twin")
    ax.bar(x + 0.27, _unit_sum(np.array(r["w_naive"])[:, -1]), 0.27, color=RED, label="naive twin (own past only)")
    ax.axhline(0.1, color=GREY, ls="--", lw=1)
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set_xlabel("walker")
    ax.set_ylabel("weight (sums to 1)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("One-way coupling: the interaction-blind average, from data alone", color=NAVY, loc="left")
    ax = axes[1]
    ax.hist(r["cos_pi_short"], bins=np.linspace(0.8, 1.0, 21), color=BLUE)
    ax.axvline(0.95, color=RED, ls="--")
    ax.set_xlabel("|cos| with the true pi")
    ax.set_ylabel("replicates")
    ax.set_title("Short data (5 000 steps), 20 runs", color=NAVY, loc="left")
    _save(fig, out, "02_linear_estimated_twin.png")


def nonlinear(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
    ku = res["kuramoto"]
    ax = axes[0]
    ax.bar(np.arange(1, 31), ku["S_test"], color=BLUE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("collective variable (rank)"), ax.set_ylabel("held-out S")
    ax.set_title(f"Kuramoto, data-only twin (most blind = mean phase, |cos| {ku['cos_target']:.3f})",
                 color=NAVY, loc="left")
    ax = axes[1]
    for key, col, lab in (("agents_align", BLUE, "alignment only"), ("agents_attract", ORANGE, "+ attraction")):
        ax.plot(np.arange(1, 13), res[key]["S_test"], "o-", color=col, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("collective variable (rank)"), ax.set_ylabel("held-out S")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Agents, data-only twin: spectrum of the 12 turns", color=NAVY, loc="left")
    _save(fig, out, "03_nonlinear_estimated_twin.png")


def fish_spectra(res, out):
    real = sorted([r for r in res["fish"] if r["kind"] == "real"], key=lambda r: r["rec"])
    virt = sorted([r for r in res["fish"] if r["kind"] == "virtual"], key=lambda r: r["rec"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), layout="constrained", sharey=True)
    k = np.arange(1, 17)
    for ax, r, v in zip(axes.flat, real, virt):
        ax.bar(k - 0.2, r["S_test"], 0.4, color=ORANGE, label="real school")
        ax.bar(k + 0.2, v["S_test"], 0.4, color=GREY, label="virtual (never met)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"8 fish #{r['rec']}", color=NAVY, fontsize=10)
        ax.set_xticks([1, 4, 8, 12, 16])
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out S")
    for ax in axes[1]:
        ax.set_xlabel("collective variable (rank)")
    fig.suptitle("Real schools: how much do interactions shape each collective variable "
                 "(8 turns + 8 speed changes)?", fontsize=12)
    _save(fig, out, "04_fish_spectra.png")


def fish_modes(res, out):
    real = sorted([r for r in res["fish"] if r["kind"] == "real"], key=lambda r: r["rec"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.6), layout="constrained", sharex=True)
    x = np.arange(16)
    w = 0.8 / len(real)
    cmap = plt.get_cmap("tab10")
    for ax, idx, title in ((axes[0], 0, "Most interaction-made collective variable"),
                           (axes[1], -1, "Most interaction-blind collective variable")):
        for i, r in enumerate(real):
            v = _signed(np.array(r["w"])[:, idx])
            ax.bar(x + (i - (len(real) - 1) / 2) * w, v, w, color=cmap(i),
                   label=f"#{r['rec']} (S = {r['S_test'][idx]:+.2f})")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(7.5, color=GREY, lw=1)
        ax.set_ylabel("weight (scaled)")
        ax.set_title(title, color=NAVY, loc="left")
        ax.legend(frameon=False, fontsize=7.5, ncols=6, loc="lower left")
    axes[1].set_xticks(x, [f"turn {i + 1}" for i in range(8)] + [f"speed {i + 1}" for i in range(8)],
                       rotation=45, ha="right", fontsize=8)
    _save(fig, out, "05_fish_modes.png")


def fish_totals(res, out):
    real = sorted([r for r in res["fish"] if r["kind"] == "real"], key=lambda r: r["rec"])
    virt = sorted([r for r in res["fish"] if r["kind"] == "virtual"], key=lambda r: r["rec"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained", sharey=True)
    x = np.arange(6)
    for ax, key, title in ((axes[0], "S_total_turn", "Group's total turn"),
                           (axes[1], "S_total_speed", "Group's total speed change")):
        ax.bar(x - 0.2, [r[key] for r in real], 0.4, color=ORANGE, label="real")
        ax.bar(x + 0.2, [v[key] for v in virt], 0.4, color=GREY, label="virtual")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [f"8 fish #{r['rec']}" for r in real], fontsize=8)
        ax.set_title(title, color=NAVY, loc="left")
    axes[0].set_ylabel("held-out S")
    axes[0].legend(frameon=False)
    _save(fig, out, "06_fish_totals.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, linear, nonlinear, fish_spectra, fish_modes, fish_totals):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e46_spectrum_estimated_twin import analyze
    d = Path("results")
    data = json.loads((d / "e46_spectrum_estimated_twin_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e46_figures", txt)
