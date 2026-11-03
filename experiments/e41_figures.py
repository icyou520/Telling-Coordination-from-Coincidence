"""Figures for e41 (and progress across e39-e41), drawn from the raw results.

    uv run python -m experiments.e41_figures      # redraw from results/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from experiments.e39_multiscale_elimination import ANTI, CANDIDATES

NAVY, BLUE, GREEN, ORANGE, RED, GREY = "#1f3a5f", "#2f7fc1", "#3a9d6a", "#e08a1e", "#c8453b", "#8a96a3"
STAT_NAMES = ["polarization", "polarization SD", "milling", "spread (cm)", "neighbour dist. (cm)"]
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _split(rows):
    syn = rows["synthetic"]
    real = [r for r in rows["real"] if r["kind"] == "real"]
    virt = [r for r in rows["real"] if r["kind"] == "virtual"]
    return syn, real, virt


def _label(r):
    return f"{r['name']} {r['truth']}" if "name" in r else f"{r['N']} fish #{r['rec']}"


def _save(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def scorecard(rows, out):
    from experiments.e41_elimination_place_speed import predictions
    preds = predictions(rows)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(preds) + 1.2))
    ax.axis("off")
    ax.set_title("e41: pre-registered predictions", loc="left", fontsize=14, color=NAVY,
                 weight="bold")
    for i, (key, label, ok, detail) in enumerate(preds):
        y = 1 - (i + 0.5) / len(preds)
        ax.add_patch(plt.Rectangle((0, y - 0.035), 0.08, 0.07, color=GREEN if ok else RED,
                                   transform=ax.transAxes))
        ax.text(0.04, y, "PASS" if ok else "FAIL", color="white", ha="center", va="center",
                weight="bold", transform=ax.transAxes)
        ax.text(0.1, y, f"{key}  {label}", va="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.1, y - 0.045, detail, va="center", transform=ax.transAxes, fontsize=8.5,
                color=GREY)
    _save(fig, out, "01_scorecard.png")


def funnel(rows, out):
    syn, real, _ = _split(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, group, title in ((axes[0], syn, "Synthetic schools (known rule)"),
                             (axes[1], real, "Real schools (Puy et al.)")):
        cmap = plt.get_cmap("tab10")
        for i, r in enumerate(group):
            ys = [25, len(r["surv0"]), len(r["surv1"]), len(r["surv2"])]
            ax.plot(range(4), ys, "o-", color=cmap(i % 10), lw=2, label=_label(r))
            ax.plot(3.7, len(r["abc_survivors"]), "x", color=cmap(i % 10), ms=8)
        ax.set_xticks([0, 1, 2, 3, 3.7], ["start", "stage 0\nfamily", "stage 1\nshort checks",
                                         "stage 2\nlong sims", "ABC-all"], fontsize=9)
        ax.set_xlim(-0.2, 4.0)
        ax.set_title(title, color=NAVY)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("candidate rules remaining")
    fig.suptitle("Elimination funnel: how many of the 25 candidate rules survive each stage "
                 "(x = survivors of the conventional search)", fontsize=11)
    _save(fig, out, "02_funnel.png")


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
            elif "surv2" not in r:
                grid[i, j] = 3
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
    ax.set_title("Where each candidate rule was eliminated (virtual groups: stages 0-1 only)",
                 color=NAVY, loc="left")
    _save(fig, out, "03_candidate_map.png")


def family_test(rows, out):
    syn, real, virt = _split(rows)
    fig, ax = plt.subplots(figsize=(12, 4))
    groups = [(syn, BLUE, "synthetic"), (real, ORANGE, "real"), (virt, GREY, "virtual")]
    x, labels = 0, []
    for g, col, name in groups:
        for r in g:
            expect = None if "truth" not in r else (r["truth"] not in ANTI)
            ax.bar(x, r["z0"], color=col, edgecolor="black" if expect else col, lw=1.5)
            labels.append(_label(r))
            x += 1
        x += 0.6
    ax.axhline(3, color=RED, ls="--", lw=1.2)
    ax.text(0, 3.2, "threshold: family eliminated above z = 3", color=RED, fontsize=9)
    pos = [i + 0.6 * (i >= len(syn)) + 0.6 * (i >= len(syn) + len(real)) for i in range(len(labels))]
    ax.set_xticks(pos, labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("z0 (total-turn test)")
    ax.set_title("Stage 0: does the pairwise structure steer the group's total turn? "
                 "(black edge: truth outside the symmetric-alignment family)", color=NAVY, loc="left",
                 fontsize=10.5)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in groups],
              [n for _, _, n in groups], frameon=False)
    _save(fig, out, "04_family_test.png")


def cost(rows, out):
    syn, real, _ = _split(rows)
    group = syn + real
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(group))
    ax.bar(x - 0.2, [r["cost_abc"] for r in group], 0.4, color=GREY, label="conventional (ABC-all)")
    ax.bar(x + 0.2, [r["cost_pruned"] for r in group], 0.4, color=BLUE, label="multi-scale (pruned)")
    for i, r in enumerate(group):
        ax.text(i + 0.2, r["cost_pruned"], f"{r['cost_pruned'] / r['cost_abc']:.2f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x, [_label(r) for r in group], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("search time (s)")
    ax.legend(frameon=False)
    ax.set_title("Search cost (numbers: pruned / conventional)", color=NAVY, loc="left")
    _save(fig, out, "05_cost.png")


def stage2(rows, out):
    _, real, _ = _split(rows)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
    for k, ax in enumerate(axes):
        for i, r in enumerate(real):
            o, s = r["obs_stats"][k], r["sim_stats"][r["shown"]][k]
            z = r["z2"][r["shown"]][k]
            ax.plot([0, 1], [i, i], color="#ddd", lw=1)
            ax.plot(0, i, "o", color="black", ms=6)
            ax.plot(1, i, "o", color=RED if z > 4 else GREEN, ms=6)
            ax.text(1.08, i, f"{o:.2f} / {s:.2f}", va="center", fontsize=7)
        ax.set_xticks([0, 1], ["observed", "simulated"])
        ax.set_xlim(-0.3, 1.9)
        ax.set_title(STAT_NAMES[k], fontsize=10, color=NAVY)
    axes[0].set_yticks(range(len(real)), [f"{_label(r)}\n({r['shown']})" for r in real], fontsize=8)
    fig.suptitle("Stage 2: long simulations of the best candidate vs the real school "
                 "(red: |z| > 4, rule eliminated on this statistic)", fontsize=11, y=1.04)
    _save(fig, out, "06_stage2_stats.png")


def schools(rows, out):
    syn, real, _ = _split(rows)
    picks = [r for r in real if (r["N"], r["rec"]) in ((39, 0), (8, 0))] + \
            [r for r in syn if r["name"] in ("T2", "T5")]
    fig, axes = plt.subplots(len(picks), 2, figsize=(9, 4.4 * len(picks)), squeeze=False,
                             layout="constrained")
    for row, r in zip(axes, picks):
        for ax, key, col, title in ((row[0], "obs", BLUE, "data"), (row[1], "sim", ORANGE,
                                                                     f"simulated: {r['shown']}")):
            ps, th = np.array(r[f"{key}_p"]), np.array(r[f"{key}_th"])
            for i in range(ps.shape[1]):
                ax.plot(ps[:, i, 0], ps[:, i, 1], color=col, alpha=0.25, lw=0.8)
            ax.quiver(ps[-1, :, 0], ps[-1, :, 1], np.cos(th[-1]), np.sin(th[-1]), color=col,
                      scale=30, width=0.006)
            ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.set_aspect("equal")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ("top", "right"):
                ax.spines[s].set_visible(True)
            ax.set_title(f"{_label(r)}: {title}", fontsize=10, color=NAVY)
    fig.suptitle("Schools in the 1 m tank: last 30 s of the data vs a long simulation", fontsize=11)
    _save(fig, out, "07_schools.png")


def timeseries(rows, out):
    _, real, _ = _split(rows)
    picks = [r for r in real if (r["N"], r["rec"]) in ((39, 0), (8, 0))]
    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 3.6 * len(picks)), squeeze=False,
                             layout="constrained")
    for row, r in zip(axes, picks):
        obs, sim = np.array(r["obs_series"]), np.array(r["sim_series"])
        for ax, k, name in ((row[0], 0, "polarization"), (row[1], 2, "spread (cm)")):
            ax.plot(np.arange(len(obs)) * 2, obs[:, k], color=BLUE, lw=0.8, label="data")
            ax.plot(np.arange(len(sim)) * 2, sim[:, k], color=ORANGE, lw=0.8,
                    label=f"simulated ({r['shown']})")
            ax.set_xlabel("time (s)"), ax.set_ylabel(name)
            ax.set_title(f"{_label(r)}: {name}", fontsize=10, color=NAVY)
            ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "08_timeseries.png")


def progress(out, results):
    runs = []
    for tag, fname in (("e39", "e39_multiscale_elimination_raw.json"),
                       ("e40", "e40_elimination_centre_twin_raw.json"),
                       ("e41", "e41_elimination_place_speed_raw.json")):
        path = results / fname
        if path.exists():
            runs.append((tag, json.loads(path.read_text())))
    if not runs:
        return
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
             "virtual groups\npassed (no false alarm)", "real schools with a\nrule passing stage 2"]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.8 / len(metrics)
    for k, (tag, vals) in enumerate(metrics.items()):
        bars = ax.bar(np.arange(5) + k * w, vals, w, label=tag, color=(GREY, ORANGE, BLUE)[k % 3])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(5) + w * (len(metrics) - 1) / 2, names, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False)
    ax.set_title("Progress across e39 (first version), e40 (+ centre pull), e41 (+ place map, speed, "
                 "adaptive stage 1)", color=NAVY, loc="left")
    _save(fig, out, "09_progress_e39_e41.png")


def summary_text(out, text):
    fig = plt.figure(figsize=(12, 0.17 * text.count("\n") + 1))
    fig.text(0.01, 0.99, text, family="monospace", fontsize=7.5, va="top")
    _save(fig, out, "10_full_results_text.png")


def truth_check(rows, out):
    syn, _, _ = _split(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    w = 0.8 / len(syn)
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(syn):
        o, s = np.array(r["obs_stats"]), np.array(r["sim_stats"][r["truth"]])
        rel = 100 * np.abs(s - o) / np.abs(o)
        x = np.arange(5) + (i - (len(syn) - 1) / 2) * w
        axes[0].bar(x, rel, w, color=cmap(i), label=_label(r))
        axes[1].bar(x, r["z2"][r["truth"]], w, color=cmap(i))
    axes[1].axhline(4, color=RED, ls="--", lw=1.2)
    axes[1].text(-0.4, 4.3, "elimination threshold z = 4", color=RED, fontsize=9)
    for ax, lab, title in ((axes[0], "mismatch (% of observed value)", "How far off is the true rule?"),
                           (axes[1], "stage-2 z", "...and how the z-test judged it")):
        ax.set_xticks(range(5), STAT_NAMES, fontsize=9)
        ax.set_ylabel(lab)
        ax.set_title(title, color=NAVY, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Stage 2 on the true rule itself (synthetic schools): small mismatches, "
                 "large z, because 30 min of simulation makes the error bars tiny", fontsize=11)
    _save(fig, out, "11_truth_stage2_check.png")


def make_figures(rows, out: Path, text: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    for f in (scorecard, funnel, candidate_map, family_test, cost, stage2, schools, timeseries,
              truth_check):
        f(rows, out)
    progress(out, out.parent)
    if text:
        summary_text(out, text)
    print(f"  figures written to {out}")


if __name__ == "__main__":
    import contextlib
    import io

    from experiments.e41_elimination_place_speed import analyze
    res = Path("results")
    data = json.loads((res / "e41_elimination_place_speed_raw.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        txt = analyze(data)
    make_figures(data, res / "e41_figures", txt)
