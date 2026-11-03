"""Figures for e43 (and progress across e39-e43), drawn from the raw results.

    uv run python -m experiments.e43_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import (BLUE, GREEN, GREY, NAVY, ORANGE, RED, _label, _save, _split, cost,
                                     family_test, funnel, plt, schools, summary_text, timeseries)
from experiments.e42_figures import candidate_map, gate, stage2, truth_check

PURPLE = "#7b5ea7"


def scorecard(rows, out):
    from experiments.e43_elimination_fish_maps import predictions
    preds = predictions(rows)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e43: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.035), 0.08, 0.07, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.04, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.1, y + 0.01, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.1, y - 0.04, detail, va="center", transform=ax.transAxes, fontsize=8.5, color=GREY)
    _save(fig, out, "01_scorecard.png")


def place_maps(rows, out):
    from experiments.e43_elimination_fish_maps import bumps
    _, real, _ = _split(rows)
    r = next(r for r in real if (r["N"], r["rec"]) == (8, 0))
    coefs = np.array(r["place_map"])
    g = np.linspace(5, 95, 13)
    gx, gy = np.meshgrid(g, g)
    b = bumps(np.stack([gx, gy], -1))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), layout="constrained")
    for i, ax in enumerate(axes.flat):
        a, c = b @ coefs[i, 5:30], b @ coefs[i, 30:55]
        psi = np.arctan2(a, c) + np.pi / 2
        strength = np.hypot(a, c)
        ax.quiver(gx, gy, strength * np.cos(psi), strength * np.sin(psi), strength, cmap="viridis",
                  scale=4 * strength.max(), width=0.008)
        ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
        ax.set_title(f"fish {i + 1}", fontsize=10, color=NAVY)
    fig.suptitle("Each fish's own place map (8 fish #0): the heading it turns toward at each spot, "
                 "ignoring the others (longer = stronger)", fontsize=11)
    _save(fig, out, "14_place_maps.png")


def virtual_compare(rows, out, results):
    from experiments.e42_elimination_cohesion_gate import gate_open, pairs
    runs = []
    prev = results / "e42_elimination_cohesion_gate_raw.json"
    if prev.exists():
        runs.append(("e42 (shared quadratic map)", json.loads(prev.read_text()), GREY))
    runs.append(("e43 (per-fish place maps)", rows, BLUE))
    names = ["'none' kept\nat stage 0", "'none' kept\nat stage 1", "'none' passes\nstage 2",
             "gate open", "real school:\na rule passes"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(runs)
    for k, (tag, data, col) in enumerate(runs):
        pr = pairs(data)
        vals = [sum("none" in v["surv0"] for _, v in pr), sum("none" in v["surv1"] for _, v in pr),
                sum(v["pass2"]["none"] for _, v in pr), sum(gate_open(v) for _, v in pr),
                sum(any(r["pass2"].values()) for r, _ in pr)]
        bars = ax.bar(np.arange(5) + k * w, vals, w, color=col, label=tag)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.1, f"{v}/8", ha="center", fontsize=9)
    ax.set_xticks(np.arange(5) + w * (len(runs) - 1) / 2, names)
    ax.set_ylim(0, 9)
    ax.set_ylabel("recordings (of 8)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("The no-interaction control: virtual groups should keep 'none' at every stage",
                 color=NAVY, loc="left")
    _save(fig, out, "15_virtual_control_e42_vs_e43.png")


def progress(out, results):
    runs = []
    for tag, fname in (("e39", "e39_multiscale_elimination_raw.json"),
                       ("e40", "e40_elimination_centre_twin_raw.json"),
                       ("e41", "e41_elimination_place_speed_raw.json"),
                       ("e42", "e42_elimination_cohesion_gate_raw.json"),
                       ("e43", "e43_elimination_fish_maps_raw.json")):
        path = results / fname
        if path.exists():
            runs.append((tag, json.loads(path.read_text())))
    metrics = {}
    for tag, rows in runs:
        syn, real, virt = _split(rows)
        inc = [r for r in syn if r["name"] != "T5"]
        metrics[tag] = [
            sum(r["truth"] in r["surv2"] for r in inc) / len(inc),
            np.mean([1 - len(r["surv1"]) / 25 for r in inc if r["name"] != "T0"]),
            sum(r["cost_pruned"] for r in inc) / sum(r["cost_abc"] for r in inc),
            sum(r["z0"] <= 3 and "none" in r["surv1"] for r in virt) / len(virt),
            sum(any(r["pass2"].values()) for r in real) / len(real),
        ]
    names = ["truth kept\n(synthetic)", "pruned by stages 0-1\n(synthetic)", "cost ratio\n(lower = faster)",
             "virtual groups: 'none'\nkept through stage 1", "real schools with a\nrule passing stage 2"]
    fig, ax = plt.subplots(figsize=(14, 4.4))
    w = 0.8 / len(metrics)
    cols = (GREY, ORANGE, BLUE, GREEN, PURPLE)
    for k, (tag, vals) in enumerate(metrics.items()):
        bars = ax.bar(np.arange(5) + k * w, vals, w, label=tag, color=cols[k % 5])
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(5) + w * (len(metrics) - 1) / 2, names, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncols=5)
    ax.set_title("Progress: e39 (first), e40 (+ centre pull), e41 (+ place map, speed), "
                 "e42 (+ tolerance, cohesion, gate), e43 (+ per-fish maps)", color=NAVY, loc="left")
    _save(fig, out, "09_progress_e39_e43.png")


def make_figures(rows, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, funnel, candidate_map, family_test, cost, stage2, schools, timeseries,
              truth_check, gate, place_maps):
        f(rows, out)
    virtual_compare(rows, out, out.parent)
    progress(out, out.parent)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e43_elimination_fish_maps import analyze
    res = Path("results")
    data = json.loads((res / "e43_elimination_fish_maps_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, res / "e43_figures", txt)
