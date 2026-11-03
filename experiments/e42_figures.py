"""Figures for e42 (and progress across e39-e42), drawn from the raw results.

    uv run python -m experiments.e42_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

from experiments.e39_multiscale_elimination import CANDIDATES
from experiments.e41_figures import (BLUE, GREEN, GREY, NAVY, ORANGE, RED, STAT_NAMES, _label, _save,
                                     _split, cost, family_test, funnel, plt, schools, summary_text,
                                     timeseries)


def scorecard(rows, out):
    from experiments.e42_elimination_cohesion_gate import predictions
    preds = predictions(rows)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e42: pre-registered predictions", loc="left", fontsize=14, color=NAVY, weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.08, 0.06, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.04, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.1, y + 0.008, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.1, y - 0.035, detail, va="center", transform=ax.transAxes, fontsize=8.5, color=GREY)
    _save(fig, out, "01_scorecard.png")


def candidate_map(rows, out):
    syn, real, virt = _split(rows)
    cols = syn + real + virt
    grid = np.zeros((len(CANDIDATES), len(cols)))
    for j, r in enumerate(cols):
        for i, h in enumerate(CANDIDATES):
            if h not in r["surv0"]:
                grid[i, j] = 0
            elif h not in r["surv1"]:
                grid[i, j] = 1
            elif h not in r["surv2"]:
                grid[i, j] = 2
            else:
                grid[i, j] = 4 if h in r["answer"] else 3
    cmap = ListedColormap(["#c8453b", "#e9a23b", "#f3d36b", "#9fd3a8", "#2e8b57"])
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    ax.set_yticks(range(len(CANDIDATES)), CANDIDATES, fontsize=8)
    ax.set_xticks(range(len(cols)), [_label(r) + (" (virtual)" if r.get("kind") == "virtual" else "")
                                     for r in cols], rotation=60, ha="right", fontsize=8)
    for j, r in enumerate(cols):
        if "truth" in r:
            i = CANDIDATES.index(r["truth"])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=2))
    for x in (len(syn) - 0.5, len(syn) + len(real) - 0.5):
        ax.axvline(x, color="white", lw=3)
    labels = ["out at stage 0", "out at stage 1", "out at stage 2", "survives", "answer"]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(5)]
              + [plt.Rectangle((0, 0), 1, 1, fill=False, ec="black", lw=2)],
              labels + ["true rule"], loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title("Where each candidate rule was eliminated (synthetic | real | virtual, all stages)",
                 color=NAVY, loc="left")
    _save(fig, out, "03_candidate_map.png")


def stage2(rows, out):
    _, real, _ = _split(rows)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
    for k, ax in enumerate(axes):
        for i, r in enumerate(real):
            o, s = r["obs_stats"][k], r["sim_stats"][r["shown"]][k]
            q = r["q2"][r["shown"]][k]
            ax.plot([0, 1], [i, i], color="#ddd", lw=1)
            ax.plot(0, i, "o", color="black", ms=6)
            ax.plot(1, i, "o", color=RED if q > 1 else GREEN, ms=6)
            ax.text(1.08, i, f"{o:.2f} / {s:.2f}", va="center", fontsize=7)
        ax.set_xticks([0, 1], ["observed", "simulated"])
        ax.set_xlim(-0.3, 1.9)
        ax.set_title(STAT_NAMES[k], fontsize=10, color=NAVY)
    axes[0].set_yticks(range(len(real)), [f"{_label(r)}\n({r['shown']})" for r in real], fontsize=8)
    fig.suptitle("Stage 2: long simulations of the best candidate vs the real school "
                 "(red: outside the tolerance)", fontsize=11, y=1.04)
    _save(fig, out, "06_stage2_stats.png")


def truth_check(rows, out):
    syn, _, _ = _split(rows)
    fig, ax = plt.subplots(figsize=(12, 4.2), layout="constrained")
    w = 0.8 / len(syn)
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(syn):
        x = np.arange(5) + (i - (len(syn) - 1) / 2) * w
        ax.bar(x, r["q2"][r["truth"]], w, color=cmap(i), label=_label(r))
    ax.axhline(1, color=RED, ls="--", lw=1.2)
    ax.text(-0.45, 1.04, "tolerance edge: above 1 = eliminated", color=RED, fontsize=9)
    ax.set_xticks(range(5), STAT_NAMES, fontsize=9)
    ax.set_ylabel("mismatch / allowed mismatch")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Stage 2 on the true rule itself (synthetic schools): T0-T4 should stay below 1, "
                 "T5 (out of class) should not", color=NAVY, loc="left")
    _save(fig, out, "11_truth_stage2_check.png")


def gate(rows, out):
    from experiments.e42_elimination_cohesion_gate import gate_open, pairs
    pr = pairs(rows)
    cols = ["virtual: 'none'\nkept at stage 0", "virtual: 'none'\nkept at stage 1",
            "virtual: 'none'\npasses stage 2", "gate open\n(virtual answer = 'none')",
            "real: a rule\npasses all stages"]
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(pr) + 1.8))
    for i, (r, v) in enumerate(pr):
        cells = ["none" in v["surv0"], "none" in v["surv1"], v["pass2"]["none"], gate_open(v),
                 bool(r["answer"]) and r["answer"] != ["none"]]
        for j, ok in enumerate(cells):
            ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, color=GREEN if ok else RED, alpha=0.85))
            ax.text(j + 0.475, i + 0.45, "yes" if ok else "no", ha="center", va="center", color="white",
                    weight="bold")
        claim = ", ".join(r["answer"]) if gate_open(v) and r["answer"] else "(no claim)"
        ax.text(5.1, i + 0.45, claim, va="center", fontsize=9)
    ax.set_xlim(0, 6.6), ax.set_ylim(len(pr), 0)
    ax.set_xticks(np.arange(5) + 0.475, cols, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(pr)) + 0.45, [_label(r) for r, _ in pr])
    ax.text(5.1, -0.08, "rule claimed", fontsize=9, weight="bold", va="bottom")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("The virtual-group gate: a real rule counts only if the no-interaction twin passes",
                 color=NAVY, loc="left", pad=40)
    _save(fig, out, "12_virtual_gate.png")


def cohesion(rows, out, results):
    _, real, _ = _split(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    ax = axes[0]
    vals = [r["cohesion_coef"]["k6-AC"][0] for r in real]
    ax.bar(range(len(real)), vals, color=[GREEN if v > 0 else RED for v in vals])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(real)), [_label(r) for r in real], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("turning cohesion coefficient (k6-AC)")
    ax.set_title("Is the pull toward the group used? (> 0 = turn toward it)", color=NAVY, loc="left")
    ax = axes[1]
    prev = results / "e41_elimination_place_speed_raw.json"
    if prev.exists():
        old = [r for r in json.loads(prev.read_text())["real"] if r["kind"] == "real"]
        ax.scatter([r["obs_stats"][3] for r in old], [r["sim_stats"][r["shown"]][3] for r in old],
                   color=GREY, s=50, label="e41 (no cohesion term)")
    ax.scatter([r["obs_stats"][3] for r in real], [r["sim_stats"][r["shown"]][3] for r in real],
               color=BLUE, s=50, label="e42 (cohesion term)")
    lim = np.array([0, 30])
    ax.plot(lim, lim, color="black", lw=1)
    ax.fill_between(lim, 0.75 * lim, 1.25 * lim, color=GREEN, alpha=0.12, label="within 25%")
    ax.set_xlim(*lim), ax.set_ylim(*lim)
    ax.set_xlabel("observed school spread (cm)"), ax.set_ylabel("simulated school spread (cm)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Do simulated schools hold together now?", color=NAVY, loc="left")
    _save(fig, out, "13_cohesion.png")


def progress(out, results):
    runs = []
    for tag, fname in (("e39", "e39_multiscale_elimination_raw.json"),
                       ("e40", "e40_elimination_centre_twin_raw.json"),
                       ("e41", "e41_elimination_place_speed_raw.json"),
                       ("e42", "e42_elimination_cohesion_gate_raw.json")):
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
    fig, ax = plt.subplots(figsize=(13, 4.2))
    w = 0.8 / len(metrics)
    for k, (tag, vals) in enumerate(metrics.items()):
        bars = ax.bar(np.arange(5) + k * w, vals, w, label=tag, color=(GREY, ORANGE, BLUE, GREEN)[k % 4])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(5) + w * (len(metrics) - 1) / 2, names, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncols=4)
    ax.set_title("Progress: e39 (first), e40 (+ centre pull), e41 (+ place map, speed), "
                 "e42 (+ tolerance, cohesion, gate)", color=NAVY, loc="left")
    _save(fig, out, "09_progress_e39_e42.png")


def make_figures(rows, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, funnel, candidate_map, family_test, cost, stage2, schools, timeseries,
              truth_check, gate):
        f(rows, out)
    cohesion(rows, out, out.parent)
    progress(out, out.parent)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e42_elimination_cohesion_gate import analyze
    res = Path("results")
    data = json.loads((res / "e42_elimination_cohesion_gate_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, res / "e42_figures", txt)
