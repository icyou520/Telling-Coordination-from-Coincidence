"""Figures for e47 (closed layers), from the raw results.

    uv run python -m experiments.e47_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREEN, GREY, NAVY, ORANGE, RED, _save, plt, summary_text

PURPLE = "#7b5ea7"
BLOCKS = (("bottom1", "global (most blind 1)", GREY), ("bottom2", "super-module layer (blind 2)", PURPLE),
          ("bottom4", "module layer (blind 4)", GREEN), ("plain_module", "plain module averages", ORANGE),
          ("top4", "4 most interaction-made", RED))


def scorecard(res, out):
    from experiments.e47_closed_layers import predictions
    preds = predictions(res)
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e47: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.07, 0.06, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.035, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.09, y + 0.012, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.09, y - 0.033, detail, va="center", transform=ax.transAxes, fontsize=8.5, color=GREY)
    _save(fig, out, "01_scorecard.png")


def spectrum(res, out):
    hr = res["hierarchy"]
    s_test, s_exact = np.array(hr["S_test"])[::-1], np.array(hr["S_exact"])[::-1]
    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    k = np.arange(1, len(s_test) + 1)
    ax.bar(k - 0.2, np.maximum(s_exact, 1e-5), 0.4, color=GREY, label="exact")
    ax.bar(k + 0.2, np.maximum(s_test, 1e-5), 0.4, color=BLUE, label="held-out estimate")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2)
    for c in hr["cuts"]:
        ax.axvline(c + 0.5, color=RED, ls="--", lw=1.2)
    for x, lab in ((1, "global"), (2, "super-\nmodules"), (3.5, "modules"), (8.5, "inside modules")):
        ax.text(x, 1.1, lab, ha="center", va="bottom", fontsize=9, color=NAVY)
    ax.set_xticks(k)
    ax.set_xlabel("collective variable, most interaction-blind first")
    ax.set_ylabel("one-step interaction share S (log)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("12 walkers in a 3-level hierarchy: the spectrum's gaps (red) are the layers, found by a fixed rule",
                 color=NAVY, loc="left")
    _save(fig, out, "02_hierarchy_spectrum.png")


def _procrustes(w, t):
    u, _, vt = np.linalg.svd(w.T @ t)
    return w @ (u @ vt)


def layer_weights(res, out):
    hr = res["hierarchy"]
    w = np.array(hr["w"])
    module = np.array(hr["module"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
    for ax, k, key, title in ((axes[0], 4, "target_module", "Module layer: discovered (bars) vs true (dots)"),
                              (axes[1], 2, "target_super", "Super-module layer: discovered vs true")):
        t = np.array(hr[key])
        wr = _procrustes(w[:, -k:], t)
        x = np.arange(len(t))
        width = 0.8 / k
        cmap = plt.get_cmap("tab10")
        for j in range(k):
            ax.bar(x + (j - (k - 1) / 2) * width, wr[:, j], width, color=cmap(j), alpha=0.8)
            ax.plot(x + (j - (k - 1) / 2) * width, t[:, j], "k.", ms=5)
        for b in np.nonzero(np.diff(module))[0]:
            ax.axvline(b + 0.5, color=GREY, lw=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [f"{i + 1}" for i in x])
        ax.set_xlabel("walker (grey lines separate modules)")
        ax.set_title(title, color=NAVY, loc="left")
    axes[0].set_ylabel("weight")
    _save(fig, out, "03_hierarchy_layers.png")


def closure(res, out):
    hr, hz = res["hierarchy"], res["horizons"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    for name, lab, col in BLOCKS:
        axes[0].plot(hz, np.maximum(hr["G"][name], 1e-5), "o-", color=col, label=lab)
        axes[1].plot(hz, hr["Sh"][name], "o-", color=col, label=lab)
    exact = hr["design"]
    for key, col in (("G_module", GREEN), ("G_plain_module", ORANGE), ("G_top4", RED)):
        axes[0].plot(hz, np.maximum(exact[key], 1e-5), ":", color=col, lw=1.5)
    axes[0].plot([], [], ":", color="black", label="exact (dotted); global exact = 0")
    axes[0].set_yscale("log"), axes[0].set_xscale("log")
    axes[0].set_ylim(1e-5, 1)
    axes[0].axhline(0.02, color=GREY, ls="--", lw=1)
    axes[0].set_xlabel("horizon h (steps)"), axes[0].set_ylabel("closure gap G (log; 0 = closed)")
    axes[0].set_title("Is the block predictable from itself?", color=NAVY, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xscale("log")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("horizon h (steps)"), axes[1].set_ylabel("interaction share at horizon h")
    axes[1].set_title("Is its future made by interaction?", color=NAVY, loc="left")
    ks = np.arange(1, len(hr["G_profile"]) + 1)
    g1 = [hr["G_profile"][f"bottom{k}"][0] for k in ks]
    axes[2].bar(ks, np.maximum(g1, 1e-5), color=BLUE)
    axes[2].set_yscale("log"), axes[2].set_ylim(1e-5, 1)
    for c in hr["cuts"]:
        axes[2].axvline(c, color=RED, ls="--", lw=1)
    axes[2].set_xlabel("block = the k most blind variables"), axes[2].set_ylabel("G at h = 1 (log)")
    axes[2].set_title("Closure of every bottom-k block", color=NAVY, loc="left")
    _save(fig, out, "04_hierarchy_closure.png")


def kuramoto(res, out):
    ku, hz = res["kuramoto"], res["horizons"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    s = np.array(ku["S_test"])[::-1]
    axes[0].bar(np.arange(1, len(s) + 1), np.maximum(s, 1e-5), color=BLUE)
    axes[0].set_yscale("log")
    for c in ku["cuts"]:
        axes[0].axvline(c + 0.5, color=RED, ls="--", lw=1.2)
    axes[0].set_xlabel("collective variable, most blind first"), axes[0].set_ylabel("one-step S (log)")
    axes[0].set_title(f"Modular Kuramoto spectrum (bottom-6 vs module order parameters: {ku['overlap']:.3f})",
                      color=NAVY, loc="left", fontsize=10)
    rg, rs = np.array(ku["G_random"]), np.array(ku["Sh_random"])
    for ax, key, rnd, ylab in ((axes[1], "G", rg, "closure gap G"), (axes[2], "Sh", rs, "interaction share at h")):
        ax.fill_between(hz, rnd.min(0), rnd.max(0), color=GREY, alpha=0.3, label="20 random 6-dim blocks")
        ax.plot(hz, np.median(rnd, 0), "-", color=GREY)
        ax.plot(hz, ku[key]["bottom6"], "o-", color=GREEN, label="discovered (6 most blind)")
        ax.plot(hz, ku[key]["target"], "x--", color=NAVY, label="module order parameters")
        ax.plot(hz, ku[key]["top6"], "o-", color=RED, label="6 most interaction-made")
        ax.set_xscale("log"), ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("horizon h (steps of 0.05)"), ax.set_ylabel(ylab)
    axes[1].set_title("Closure", color=NAVY, loc="left")
    axes[2].set_title("Made by interaction?", color=NAVY, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, out, "05_kuramoto_layer.png")


def layer_map(res, out):
    hr, ku = res["hierarchy"], res["kuramoto"]
    fig, ax = plt.subplots(figsize=(8.5, 6.2), layout="constrained")

    def pt(g, sh, col, lab, marker="o"):
        ax.scatter(max(max(g), 1e-4), max(sh), s=90, color=col, marker=marker, label=lab, zorder=3)

    for name, lab, col in BLOCKS:
        pt(hr["G"][name], hr["Sh"][name], col, "walkers: " + lab)
    pt(ku["G"]["bottom6"], ku["Sh"]["bottom6"], GREEN, "oscillators: discovered layer", "s")
    pt(ku["G"]["top6"], ku["Sh"]["top6"], RED, "oscillators: 6 most interaction-made", "s")
    for g, s in zip(ku["G_random"], ku["Sh_random"]):
        ax.scatter(max(max(g), 1e-4), max(s), s=20, color=GREY, marker="s", alpha=0.6)
    ax.scatter([], [], s=20, color=GREY, marker="s", label="oscillators: random blocks")
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 1)
    ax.set_ylim(-0.05, 1.0)
    ax.axvline(0.05, color=GREY, ls="--", lw=1)
    ax.axhline(0.02, color=GREY, ls="--", lw=1)
    kw = {"fontsize": 9, "transform": ax.transAxes}
    ax.text(0.02, 0.97, "EMERGENT LAYER\nclosed, made by interaction", color=GREEN, va="top", **kw)
    ax.text(0.02, 0.09, "INHERITED\nclosed, untouched", color=GREY, va="bottom", **kw)
    ax.text(0.60, 0.97, "NOT A LAYER\nneeds the parts", color=RED, va="top", **kw)
    ax.set_xlabel("largest closure gap over h = 1-100 (log; simulation estimates,\n"
                  "biased up by ~0.01-0.02 at h = 100 from finite data)")
    ax.set_ylabel("largest interaction share over horizons")
    ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.45))
    ax.set_title("Which blocks of collective variables are layers?", color=NAVY, loc="left")
    _save(fig, out, "06_layer_map.png")


def make_figures(res, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, spectrum, layer_weights, closure, kuramoto, layer_map):
        f(res, out)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e47_closed_layers import analyze
    d = Path("results")
    data = json.loads((d / "e47_closed_layers_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, d / "e47_figures", txt)
