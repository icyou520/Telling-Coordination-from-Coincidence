"""Figures for e84 (can memory find primes?).

    uv run python -m experiments.e84_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard


def _power(x):
    m, e = f"{x:.0e}".split("e")
    return f"10^{int(e)}" if m == "1" else f"{m}x10^{int(e)}"


def _ranges(res):
    return [(f"{_power(res['dev']['x'])} (development)", res["dev"])] + [(_power(r["x"]), r) for r in res["tests"]]


def search(res, out):
    rows = _ranges(res)
    labels = ["memory (frozen)", "memory (refitted)", "oracle: true block counts", "trial division to 1000"]
    keys = ["search_memory", "search_crossfit", "search_oracle", "deep_filter"]
    colors = [BLUE, BLUE, GREY, ORANGE]
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    w = 0.8 / len(rows)
    for j, (name, r) in enumerate(rows):
        v = [100 * r["real"][k] for k in keys]
        ax.barh(np.arange(len(keys)) + j * w, v, height=w, color=colors, alpha=1 - 0.25 * j,
                edgecolor="black", lw=0.4)
        for i, x in enumerate(v):
            ax.text(max(x, 0) + 0.5, i + j * w, f"{name}: {x:+.2f}%", va="center", fontsize=7)
    ax.set_yticks(np.arange(len(keys)) + 0.4 - w / 2, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("more primes per primality test than testing every survivor of trial division to 101 (%)")
    ax.set_xlim(right=75)
    ax.set_title("Where to look: memory against divisibility", color=NAVY, loc="left", fontsize=10)
    _save(fig, out, "02_search_gain.png")


def memory(res, out):
    rows = _ranges(res)
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 3.6), layout="constrained", sharey=True)
    lags = np.arange(1, 11)
    for ax, (name, r) in zip(np.atleast_1d(axes), rows):
        band = 2 / np.sqrt(r["blocks"])
        ax.axhspan(-band, band, color=GREY, alpha=0.2, lw=0)
        ax.plot(lags, r["real"]["acf"], "o-", color=BLUE, ms=4, label="primes")
        ax.plot(lags, r["sieve_random"]["acf"], "x--", color=GREY, ms=4, label="sieve-random")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{name}\nvariance/Poisson: primes {r['real']['var_ratio']:.2f}, "
                     f"sieve-random {r['sieve_random']['var_ratio']:.2f}", color=NAVY, fontsize=9, loc="left")
        ax.set_xlabel("lag (blocks)")
    np.atleast_1d(axes)[0].set_ylabel("autocorrelation of block surplus z")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    _save(fig, out, "03_block_memory.png")


def shares(res, out):
    rows = _ranges(res)
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    kinds = [("real", "primes", BLUE), ("sieve_random", "sieve-random", GREY), ("planted", "planted signal", ORANGE)]
    for j, (name, r) in enumerate(rows):
        for i, (k, lab, c) in enumerate(kinds):
            t = r[k]["twinshare"]
            x = j + (i - 1) * 0.22
            ax.errorbar(x, t["S"], yerr=[[t["S"] - t["interval"][0]], [t["interval"][1] - t["S"]]], fmt="o",
                        color=c, ms=5, capsize=3, label=lab if j == 0 else None)
            ax.plot([x - 0.08, x + 0.08], [t["null_bar"]] * 2, color="black", lw=1)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(rows)), [n for n, _ in rows])
    ax.set_ylabel("interaction share S (twinshare)")
    ax.set_title("Digit classes 1/3/7/9 as members (black ticks: no-coordination bar)", color=NAVY, loc="left",
                 fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, out, "04_twinshare.png")


def make_figures(res, out, text=None):
    from experiments.e84_primes import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e84: pre-registered predictions", out)
    search(res, out)
    memory(res, out)
    shares(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e84_primes_raw.json").read_text()), Path("results/e84_figures"))
