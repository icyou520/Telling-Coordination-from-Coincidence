"""Figures for e44 (and progress across e39-e44), drawn from the raw results.

    uv run python -m experiments.e44_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e41_figures import (BLUE, GREEN, GREY, NAVY, ORANGE, RED, _label, _save, _split, cost,
                                     family_test, funnel, plt, schools, summary_text, timeseries)
from experiments.e42_figures import candidate_map, gate, stage2, truth_check
from experiments.e43_figures import PURPLE, place_maps

TEAL = "#2a9d8f"
RUNS = (("e39", "e39_multiscale_elimination_raw.json"),
        ("e40", "e40_elimination_centre_twin_raw.json"),
        ("e41", "e41_elimination_place_speed_raw.json"),
        ("e42", "e42_elimination_cohesion_gate_raw.json"),
        ("e43", "e43_elimination_fish_maps_raw.json"),
        ("e44", "e44_elimination_speed_heading_raw.json"))


def scorecard(rows, out):
    from experiments.e44_elimination_speed_heading import predictions
    preds = predictions(rows)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e44: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.08, 0.06, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.04, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.1, y + 0.01, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.1, y - 0.035, detail, va="center", transform=ax.transAxes, fontsize=8.5, color=GREY)
    _save(fig, out, "01_scorecard.png")


def mechanism(rows, out, results):
    """The e43b diagnosis, before (e43 model) and after (e44 model)."""
    prev = results / "e43b_virtual_diagnostic_raw.json"
    after = sorted([r for r in rows["real"] if r["N"] == 8 and r["kind"] == "virtual"], key=lambda r: r["rec"])
    old = {r["rec"]: r for r in json.loads(prev.read_text()) if r["kind"] == "virtual"} if prev.exists() else {}
    before = [old[r["rec"]] for r in after] if all(r["rec"] in old for r in after) else []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    x = np.arange(len(after))
    w = 0.38
    ax = axes[0]
    if before:
        ax.bar(x - w / 2, [r["none@10"]["bias"][2] for r in before], w, color=GREY,
               label="e43 model (speed map without heading)")
    ax.bar(x + w / 2, [r["bias"]["none@10"][2] for r in after], w, color=BLUE,
           label="e44 model (speed map with heading)")
    ax.axhspan(-0.5, 0.5, color=GREEN, alpha=0.12)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f"virtual #{r['rec']}" for r in after])
    ax.set_ylabel("spread forecast bias at 1 s (cm)")
    ax.set_title("No-interaction model: does it still drift apart too fast?", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    if before:
        ax.bar(x - w / 2, [r["coef_k6C_speed"][1] for r in before], w, color=GREY, label="e43 model")
    ax.bar(x + w / 2, [r["bias"]["k6C_speed_coef"][1] for r in after], w, color=BLUE, label="e44 model")
    ax.axhspan(-0.1, 0.1, color=GREEN, alpha=0.12)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f"virtual #{r['rec']}" for r in after])
    ax.set_ylabel("k6-C speed cohesion coefficient")
    ax.set_title("Fake 'speed up toward the group' in fish that never met", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Mechanism check on the 6 virtual 8-fish groups (green band: pre-registered target)",
                 fontsize=11)
    _save(fig, out, "16_mechanism.png")


def virtual_compare(rows, out, results):
    from experiments.e42_elimination_cohesion_gate import gate_open, pairs
    runs = []
    for tag, fname, col in (("e42 (shared quadratic map)", RUNS[3][1], GREY),
                            ("e43 (per-fish turning maps)", RUNS[4][1], PURPLE)):
        path = results / fname
        if path.exists():
            runs.append((tag, json.loads(path.read_text()), col))
    runs.append(("e44 (+ speed maps with heading)", rows, TEAL))
    names = ["'none' kept\nat stage 0", "'none' kept\nat stage 1", "'none' passes\nstage 2",
             "gate open", "real school:\na rule passes"]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.8 / len(runs)
    for k, (tag, data, col) in enumerate(runs):
        pr = pairs(data)
        vals = [sum("none" in v["surv0"] for _, v in pr), sum("none" in v["surv1"] for _, v in pr),
                sum(v["pass2"]["none"] for _, v in pr), sum(gate_open(v) for _, v in pr),
                sum(any(r["pass2"].values()) for r, _ in pr)]
        bars = ax.bar(np.arange(5) + k * w, vals, w, color=col, label=tag)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.1, f"{v}/8", ha="center", fontsize=8.5)
    ax.set_xticks(np.arange(5) + w * (len(runs) - 1) / 2, names)
    ax.set_ylim(0, 9.5)
    ax.set_ylabel("recordings (of 8)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("The no-interaction control: virtual groups should keep 'none' at every stage",
                 color=NAVY, loc="left")
    _save(fig, out, "15_virtual_control_e42_to_e44.png")


def progress(out, results):
    runs = [(tag, json.loads((results / f).read_text())) for tag, f in RUNS if (results / f).exists()]
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
    w = 0.84 / len(metrics)
    cols = (GREY, ORANGE, BLUE, GREEN, PURPLE, TEAL)
    for k, (tag, vals) in enumerate(metrics.items()):
        bars = ax.bar(np.arange(5) + k * w, vals, w, label=tag, color=cols[k % 6])
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(np.arange(5) + w * (len(metrics) - 1) / 2, names, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncols=6)
    ax.set_title("Progress e39 -> e44: + centre pull, + place map & speed, + tolerance/cohesion/gate, "
                 "+ per-fish maps, + heading-aware speed", color=NAVY, loc="left", fontsize=10.5)
    _save(fig, out, "09_progress_e39_e44.png")


def make_figures(rows, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, funnel, candidate_map, family_test, cost, stage2, schools, timeseries,
              truth_check, gate, place_maps):
        f(rows, out)
    mechanism(rows, out, out.parent)
    virtual_compare(rows, out, out.parent)
    progress(out, out.parent)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e44_elimination_speed_heading import analyze
    res = Path("results")
    data = json.loads((res / "e44_elimination_speed_heading_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, res / "e44_figures", txt)
