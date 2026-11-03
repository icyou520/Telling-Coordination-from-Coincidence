"""Figures for e45, the interaction-share spectrum, drawn from the raw results.

    uv run python -m experiments.e45_figures      # redraw from results/*.json
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


def scorecard(res, out):
    from experiments.e45_interaction_spectrum import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e45: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.07, 0.06, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.035, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.09, y + 0.012, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.09, y - 0.033, detail, va="center", transform=ax.transAxes, fontsize=8.5, color=GREY)
    _save(fig, out, "01_scorecard.png")


def discovered_blind(res, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    x = np.arange(10)
    for ax, key, title in ((axes[0], "L1", "Walkers on a chain"), (axes[1], "L2", "Walkers, all-to-all"),
                           (axes[2], "L3", "One-way coupling (directed graph)")):
        r = res[key]
        w = _unit_sum(np.array(r["w"])[:, -1])
        ax.bar(x - 0.2, w, 0.4, color=BLUE, label="discovered (most interaction-blind)")
        truth = _unit_sum(r["pi"]) if "pi" in r else np.full(10, 0.1)
        ax.bar(x + 0.2, truth, 0.4, color=GREEN,
               label="theory: left null vector " + ("pi" if "pi" in r else "(centre of mass)"))
        ax.axhline(0.1, color=GREY, ls="--", lw=1)
        ax.text(9.4, 0.103, "plain average", color=GREY, fontsize=8, ha="right")
        ax.set_xticks(x, [str(i + 1) for i in x])
        ax.set_xlabel("walker")
        ax.set_title(title, color=NAVY, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("weight in the collective variable (sums to 1)")
    fig.suptitle("The collective variable interactions cannot touch, found from data alone", fontsize=12)
    _save(fig, out, "02_discovered_blind_variable.png")


def spectra(res, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained", sharey=True)
    for ax, key, title in ((axes[0], "L1", "Chain"), (axes[1], "L2", "All-to-all"), (axes[2], "L3", "Directed")):
        r = res[key]
        k = np.arange(1, 11)
        ax.bar(k, r["S_test"], color=BLUE, label="discovered, held-out S")
        ax.plot(k, r["S_exact"], "o", color=ORANGE, label="exact S")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(k)
        ax.set_xlabel("collective variable (rank)")
        ax.set_title(title, color=NAVY, loc="left")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("interaction share S")
    fig.suptitle("The full spectrum: every collective variable, from most interaction-made (left) "
                 "to interaction-blind (right)", fontsize=12)
    _save(fig, out, "03_spectra_linear.png")


def chain_modes(res, out):
    r = res["L1"]
    w = np.array(r["w"])
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    x = np.arange(1, 11)
    for k, col in ((0, RED), (1, ORANGE), (2, PURPLE)):
        v = w[:, k] / np.abs(w[:, k]).max()
        ax.plot(x, v, "o-", color=col, label=f"rank {k + 1}: S = {r['S_test'][k]:.2f}")
    v = w[:, -1] / np.abs(w[:, -1]).max()
    ax.plot(x, v * np.sign(v.sum()), "s-", color=BLUE, label=f"most blind: S = {r['S_test'][-1]:+.3f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("walker position along the chain")
    ax.set_ylabel("weight (scaled)")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.set_title("Chain walkers: the most interaction-made variables found are the zig-zag "
                 "(shortest-wavelength) modes", color=NAVY, loc="left")
    _save(fig, out, "04_chain_modes.png")


def graph(res, out):
    r = res["L3"]
    a = np.array(r["A"])
    pi = _unit_sum(r["pi"])
    n = len(a)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = np.stack([np.cos(ang), np.sin(ang)], 1)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    off = np.abs(a - np.diag(np.diag(a)))
    for i in range(n):
        for j in range(n):
            if i != j and off[i, j] > 0:
                ax.annotate("", xy=pos[j] * 0.9, xytext=pos[i] * 0.9,
                            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.5 + 6 * off[i, j] / off.max(),
                                            alpha=0.7, shrinkA=12, shrinkB=12))
    ax.scatter(pos[:, 0] * 0.9, pos[:, 1] * 0.9, s=6000 * pi, color=BLUE, zorder=3)
    for i in range(n):
        ax.text(pos[i, 0] * 1.08, pos[i, 1] * 1.08, f"{i + 1}\n{pi[i]:.2f}", ha="center", va="center",
                fontsize=8)
    ax.set_xlim(-1.3, 1.3), ax.set_ylim(-1.3, 1.3), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title("One-way coupling: arrow i -> j means i is pulled toward j\n"
                 "(node size = weight in the interaction-blind average)", color=NAVY, fontsize=10)
    _save(fig, out, "05_directed_graph.png")


def kuramoto(res, out):
    r = res["N1"]
    w = np.array(r["w"])[:, -1]
    w = w / w[:10].sum()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
    ax = axes[0]
    ax.bar(np.arange(30), w, color=[BLUE] * 10 + [ORANGE] * 10 + [PURPLE] * 10)
    ax.set_xticks([4.5, 14.5, 24.5], ["phases theta_i", "cos theta_i", "sin theta_i"])
    ax.axhline(0.1, color=GREY, ls="--", lw=1)
    ax.set_ylabel("weight")
    ax.set_title(f"Kuramoto: most interaction-blind variable (S = {r['S_test'][-1]:+.3f}) = mean phase",
                 color=NAVY, loc="left")
    ax = axes[1]
    ax.bar(np.arange(1, 31), r["S_test"], color=BLUE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("collective variable (rank)")
    ax.set_ylabel("held-out S")
    ax.set_title("Full spectrum over 30 candidate observables", color=NAVY, loc="left")
    _save(fig, out, "06_kuramoto.png")


def agents(res, out):
    al, at = res["N2-align"], res["N2-attract"]
    fig = plt.figure(figsize=(15, 4.6), layout="constrained")
    gs = fig.add_gridspec(1, 4)
    for k, (r, title) in enumerate(((al, "alignment only"), (at, "alignment + attraction"))):
        ax = fig.add_subplot(gs[0, k])
        s = r["snaps"][-1]
        p, th = np.array(s["p"]), np.array(s["th"])
        ax.quiver(p[:, 0], p[:, 1], np.cos(th), np.sin(th), color=BLUE if k == 0 else ORANGE, scale=18)
        ax.set_xlim(0, r["box"]), ax.set_ylim(0, r["box"]), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(True)
        ax.set_title(title, color=NAVY, fontsize=10)
    ax = fig.add_subplot(gs[0, 2])
    ax.bar([0, 1], [al["S_total_test"], at["S_total_test"]], color=[BLUE, ORANGE])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([0, 1], ["alignment\nonly", "+ attraction"])
    ax.set_ylabel("held-out S of the total turn")
    ax.set_title("Is the group's total turn\ninteraction-blind?", color=NAVY, fontsize=10)
    ax = fig.add_subplot(gs[0, 3])
    for r, col, lab in ((al, BLUE, "alignment only"), (at, ORANGE, "+ attraction")):
        ax.plot(np.arange(1, 13), r["S_test"], "o-", color=col, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("collective variable (rank)")
    ax.set_ylabel("held-out S")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Spectrum of the 12 turns", color=NAVY, fontsize=10)
    _save(fig, out, "07_agents.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, discovered_blind, spectra, chain_modes, graph, kuramoto, agents):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e45_interaction_spectrum import analyze
    res_dir = Path("results")
    data = json.loads((res_dir / "e45_interaction_spectrum_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, res_dir / "e45_figures", txt)
