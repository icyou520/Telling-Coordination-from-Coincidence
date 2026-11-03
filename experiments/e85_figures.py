"""Figures for e85 (memory on top of every filter).

    uv run python -m experiments.e85_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import BLUE, GREY, NAVY, ORANGE, _save, plt, summary_text
from experiments.e71_figures import scorecard
from experiments.e84_figures import _power


def _ranges(res):
    return [(f"{_power(res['dev']['x'])} (development)", res["dev"])] + [(_power(r["x"]), r) for r in res["tests"]]


def gains(res, out):
    rows = _ranges(res)
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.8), layout="constrained", sharey=True)
    for ax, (name, r) in zip(np.atleast_1d(axes), rows):
        d = [int(k) for k in r["depths"]]
        m = list(r["depths"].values())
        g = lambda k: 100 * np.array([v[k] for v in m])  # noqa: E731
        sd = 3 * np.maximum(g("random_sd"), g("charged_sd"))
        ax.fill_between(d, -sd, sd, color=GREY, alpha=0.25, lw=0, label="chance (3 s.d.)")
        ax.plot(d, g("oracle"), "s-", color=GREY, label="oracle ceiling")
        ax.plot(d, g("free"), "o-", color=BLUE, label="memory, free information")
        ax.plot(d, g("charged"), "^--", color=ORANGE, label="memory, information charged")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("trial division depth D")
        ax.set_title(name, color=NAVY, loc="left", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("extra primes per test (%)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _save(fig, out, "02_memory_by_depth.png")


def tests(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for (name, r), c in zip(_ranges(res), (GREY, BLUE, ORANGE)):
        d = [int(k) for k in r["depths"]]
        m = list(r["depths"].values())
        axes[0].plot(d, [v["tests_per_prime"] for v in m], "o-", color=c, label=name)
        axes[1].plot(d, [v["calibration"] for v in m], "o-", color=c, label=name)
    axes[0].set_ylabel("primality tests per prime found")
    axes[0].set_title("Each filter step saves tests", color=NAVY, loc="left", fontsize=10)
    axes[1].axhline(1, color="black", lw=0.6)
    axes[1].set_ylabel("primes / Mertens baseline")
    axes[1].set_title("Near sqrt(x) the baseline needs Buchstab's correction", color=NAVY, loc="left", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("trial division depth D")
        ax.legend(fontsize=8)
    _save(fig, out, "03_filters.png")


def make_figures(res, out, text=None):
    from experiments.e85_prime_filters import predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard(predictions(res), "e85: pre-registered predictions", out)
    gains(res, out)
    tests(res, out)
    if text:
        summary_text(out, text)


if __name__ == "__main__":
    make_figures(json.loads(Path("results/e85_prime_filters_raw.json").read_text()), Path("results/e85_figures"))
