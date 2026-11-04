"""Build the white paper PDF: figures from saved results, then markdown to PDF via headless Chrome.

    uv run --with markdown python whitepaper/build.py
"""

import json
import subprocess
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

AUTHOR = "Eric J Couture"
TITLE = "Telling Coordination from Coincidence"
SUBTITLE = "The interaction share: a test for when a group\u2019s behaviour is made by its members\u2019 interactions"
DATE = "October 2026"

NAVY, BLUE, GREEN, ORANGE, GREY = "#1f3a5f", "#2f7fc1", "#3a9d6a", "#e08a1e", "#7a8694"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555", "svg.fonttype": "none"})


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    fig.savefig(FIG / name.replace(".svg", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_twin():
    rng = np.random.default_rng(4)
    pos = rng.normal(0, 1, (60, 2)) * [1.4, 0.9]
    aligned = 0.5 + rng.normal(0, 0.25, 60)
    loose = rng.uniform(-np.pi, np.pi, 60)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, th, col, title in [(axes[0], aligned, BLUE, "Real group: members interact"),
                               (axes[1], loose, GREEN, "Twin: same members, interactions off")]:
        ax.quiver(pos[:, 0], pos[:, 1], np.cos(th), np.sin(th), color=col, scale=22, width=0.006,
                  pivot="middle")
        phi = abs(np.exp(1j * th).mean())
        ax.set_title(title, fontsize=11, color=NAVY)
        ax.text(0.5, -0.06, f"alignment {phi:.0%}", transform=ax.transAxes, ha="center",
                color=GREY)
        ax.set_xlim(-4.2, 4.2), ax.set_ylim(-3, 3), ax.set_aspect("equal"), ax.axis("off")
    save(fig, "fig1_twin.svg")


def fig_walkers():
    theta, n = 0.1, 10
    lam = 2 - 2 * np.cos(9 * np.pi / n)
    g = np.linspace(0, 0.4, 200)
    a, b = 1 - theta - g * lam, 1 - theta
    s_mode = (a - b) ** 2 / ((1 - a ** 2) + (a - b) ** 2)
    gp = [0, 0.1, 0.2, 0.4]
    fig, (l, r) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    l.plot(g, np.zeros_like(g), color=GREEN, lw=2.2, label="average position")
    l.plot(g, s_mode, color=BLUE, lw=2.2, label="fastest vibration pattern")
    l.set_xlabel("spring strength"), l.set_ylabel("interaction share S")
    l.set_ylim(-0.05, 1), l.legend(frameon=False, loc="upper left")
    l.set_title("Interaction share", fontsize=11, color=NAVY)
    r.axhline(0, color="#bbb", lw=0.8)
    r.plot(gp, [0.41, -0.03, -0.25, -0.35], "o-", color=GREEN, lw=2, label="average position")
    r.plot(gp, [0.40, 0.08, 0.00, 0.08], "o-", color=BLUE, lw=2, label="fastest vibration pattern")
    r.set_xlabel("spring strength"), r.set_ylabel("\u03a8 (nats)")
    r.legend(frameon=False, loc="upper right")
    r.set_title("Established measure \u03a8", fontsize=11, color=NAVY)
    fig.tight_layout()
    save(fig, "fig2_walkers.svg")


def fig_transitions():
    bj = [0, 0.5, 0.9, 1.0, 1.1, 1.25, 1.5]
    s_cw = [0.0, 0.077, 0.443, 0.597, 0.743, 0.893, 0.973]
    ks = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    s_k = [0.0026, 0.0812, 0.3163, 0.8687, 0.9933, 0.9989, 0.9994]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, x, y, xc, title in [(axes[0], bj, s_cw, 1.0, "Magnet (Curie\u2013Weiss)"),
                                (axes[1], ks, s_k, 1.5, "Oscillators (Kuramoto)")]:
        ax.plot(x, y, "o-", color=BLUE, lw=2.2)
        ax.axvline(xc, color=GREY, ls="--", lw=1)
        ax.text(xc, 0.04, " order appears", color=GREY, fontsize=9)
        ax.set_ylim(-0.03, 1.03), ax.set_xlabel("coupling strength")
        ax.set_ylabel("interaction share"), ax.set_title(title, fontsize=11, color=NAVY)
    fig.tight_layout()
    save(fig, "fig3_transitions.svg")


def fig_fish():
    puy = json.loads((ROOT / "results/e33_fish_projected_twin_raw.json").read_text())["real"]
    zf = json.loads((ROOT / "results/e38_zebrafish_raw.json").read_text())
    groups = [("real", [r["real"]["S_proj"] for r in puy], BLUE, 0),
              ("virtual", [r["virtual"]["S_proj"] for r in puy], GREEN, 1),
              ("real", [r["real"]["S_proj"] for r in zf["real"]], BLUE, 2.6),
              ("virtual", [r["virtual"]["S_proj"] for r in zf["virtual"]], GREEN, 3.6)]
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.axhline(0, color="#bbb", lw=0.8)
    for _, vals, col, x0 in groups:
        ax.scatter(x0 + rng.uniform(-0.17, 0.17, len(vals)), vals, s=42, color=col,
                   edgecolor="white", lw=0.8, zorder=3)
    ax.set_xticks([g[3] for g in groups], [g[0] for g in groups])
    ax.set_xlim(-0.6, 4.2), ax.set_ylim(-0.15, 1.0)
    ax.set_ylabel("projected interaction share")
    for x, label in [(0.5, "Adult fish, 8\u201339 per group"), (3.1, "Zebrafish, 60\u2013100 per group")]:
        ax.text(x, -0.2, label, transform=ax.get_xaxis_transform(), ha="center", color=GREY)
    save(fig, "fig4_fish.svg")


def fig_closure():
    runs = json.loads((ROOT / "results/e51_trusted_closure_raw.json").read_text())["runs"]
    real = sorted([r for r in runs if r["N"] == 8 and r["kind"] == "real"], key=lambda r: r["rec"])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axhline(0.05, color=GREY, ls="--", lw=1)
    ax.text(2.45, 0.053, "planted signal", color=GREY, fontsize=8, ha="right", va="bottom")
    for x, (key, label) in enumerate((("shape", "shape"), ("motion", "motion"), ("turn", "next turn"))):
        col = ORANGE if key == "turn" else BLUE
        for j, r in enumerate(real):
            g = r["h"]["10"][key]["conf"]
            xx = x + (j - 2.5) * 0.07
            ax.errorbar(xx, g["G"], yerr=[[g["G"] - g["lo"]], [g["hi"] - g["G"]]], fmt="o", color=col, ms=5,
                        lw=1, capsize=0)
    ax.set_xticks([0, 1, 2], ["shape\n(alignment, spread, milling)", "motion\n(centre, heading, speed)",
                              "next turn\n(heading change)"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("extra from the members, 1 s ahead")
    save(fig, "fig5_closure.svg")


def fig_influence():
    e52 = json.loads((ROOT / "results/e52_influence_map_raw.json").read_text())["runs"]
    e53 = json.loads((ROOT / "results/e53_influence_replication_raw.json").read_text())["runs"]
    panels = [("Adult fish, real (8)", [r for r in e52 if r["kind"] == "real"]),
              ("Adult fish, virtual (6)", [r for r in e52 if r["kind"] == "virtual"]),
              ("Zebrafish, real (9)", [r for r in e53 if r["data"] == "zebrafish" and r["kind"] == "real"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, (title, rows) in zip(axes, panels):
        m = np.mean([np.array(r["maps"]["sin"]) / np.abs(np.array(r["maps"]["sin"])).max() for r in rows], 0)
        im = ax.imshow(m, origin="lower", extent=[-2.5, 2.5, -2.5, 2.5], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.annotate("", xy=(2.2, -2.1), xytext=(0.9, -2.1), arrowprops=dict(arrowstyle="->", color=NAVY))
        ax.text(0.9, -1.85, "swimming", color=NAVY, fontsize=7)
        ax.set_title(title, fontsize=10, color=NAVY)
        ax.set_xticks([-2, 0, 2], ["back", "centre", "front"], fontsize=8)
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["right turn", "0", "left turn"], fontsize=8)
    save(fig, "fig6_influence.svg")


def fig_open():
    e54 = json.loads((ROOT / "results/e54_neutral_directions_raw.json").read_text())["runs"]
    e55 = json.loads((ROOT / "results/e55_free_members_raw.json").read_text())["runs"]
    e57 = json.loads((ROOT / "results/e57_oscillator_looseness_raw.json").read_text())["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    fl = sorted([r for r in e54 if r["system"] == "flock"], key=lambda r: r["field"])
    fields = sorted({r["field"] for r in fl})
    axes[0].plot([fields.index(r["field"]) for r in fl], [r["turn"]["arrangement"]["G"] for r in fl], "o",
                 color=ORANGE, ms=5)
    axes[0].set_xticks(range(len(fields)), [f"{f:g}" for f in fields], fontsize=8)
    axes[0].set_xlabel("field pinning the direction", fontsize=9)
    axes[0].set_ylabel("closure gap G", fontsize=9)
    axes[0].set_title("Flock: pinning does\nnot close the turn", fontsize=9.5, color=NAVY)
    ku = [r for r in e55 if r["system"] == "kuramoto"]
    ms = sorted({r["m"] for r in ku})
    axes[1].plot([ms.index(r["m"]) for r in ku], [r["stable"]["arrangement"]["G"] for r in ku], "o",
                 color=BLUE, ms=5)
    axes[1].set_xticks(range(len(ms)), [str(m) for m in ms], fontsize=8)
    axes[1].set_xlabel("free oscillators (of 30)", fontsize=9)
    axes[1].set_title("Oscillators: one free\nmember opens synchrony", fontsize=9.5, color=NAVY)
    axes[2].plot([r["max_lam"] for r in e57], [r["r"]["loosest"]["G"] for r in e57], "o", color=BLUE, ms=4.5,
                 label="3 loosest members")
    axes[2].plot([r["max_lam"] for r in e57], [r["r"]["tightest"]["G"] for r in e57], "x", color=GREY, ms=4.5,
                 label="3 tightest members")
    axes[2].set_xlabel("looseness of loosest member", fontsize=9)
    axes[2].set_title("30 random oscillator groups:\nthe loose members carry it", fontsize=9.5, color=NAVY)
    axes[2].legend(frameon=False, fontsize=7, loc="center left")
    for ax in axes:
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig7_open.svg")


def fig_brain():
    e62 = json.loads((ROOT / "results/e62_neural_share_repaired_raw.json").read_text())
    e63 = json.loads((ROOT / "results/e63_area_closure_raw.json").read_text())["real"]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    conds = [("independent", "none"), ("weak", "weak\ncoupling"), ("coupled", "realistic\ncoupling"),
             ("hidden", "hidden\ndrive"), ("recorded", "drive\nrecorded")]
    for k, (c, label) in enumerate(conds):
        v = [r["E0"]["S"] for r in e62["sim"] if r["cond"] == c]
        axes[0].plot(np.full(len(v), k), v, "o", color=ORANGE if c == "hidden" else BLUE, ms=4, alpha=0.8)
    axes[0].set_xticks(range(len(conds)), [lab for _, lab in conds], fontsize=7.5)
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set_yticks([0, 0.01, 0.1, 0.4], ["0", "0.01", "0.1", "0.4"])
    axes[0].set_ylim(-0.004, 0.6)
    axes[0].set_ylabel("interaction share", fontsize=9)
    axes[0].set_title("Simulated networks:\nknown answers", fontsize=9.5, color=NAVY)
    real = e62["real"]
    data = [[r[k]["S"] for r in real] for k in ("E0", "E1", "E2")] + [[r["virtual"]["S"] for r in real]]
    b = axes[1].boxplot(data, showfliers=False, widths=0.55, patch_artist=True)
    for p, c in zip(b["boxes"], (BLUE, BLUE, BLUE, GREEN)):
        p.set_facecolor(c), p.set_alpha(0.5)
    axes[1].set_xticks([1, 2, 3, 4], ["stimulus", "+ task,\nbehaviour", "+ other\nareas", "virtual"], fontsize=7.5)
    axes[1].set_title("221 mouse brain areas:\ninteraction share", fontsize=9.5, color=NAVY)
    for k, (extra, c, label) in enumerate((("internal", BLUE, "rest of the area"), ("other", ORANGE, "other areas"))):
        med = [np.median([x[f"state_{h}_{extra}"]["G"] for x in e63 if f"state_{h}_{extra}" in x]) for h in (5, 10, 20)]
        axes[2].plot([50, 100, 200], med, "o-", color=c, lw=2, label=label)
    axes[2].set_xlabel("forecast horizon (ms)", fontsize=9)
    axes[2].set_ylabel("closure gap G (median)", fontsize=9)
    axes[2].set_ylim(0, 0.016)
    axes[2].legend(frameon=False, fontsize=7.5)
    axes[2].set_title("Is the area's summary\na closed level?", fontsize=9.5, color=NAVY)
    for ax in axes:
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig8_brain.svg")


def fig_markets():
    e66 = json.loads((ROOT / "results/e66_market_volatility_raw.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), gridspec_kw={"width_ratios": [1, 1.5]})
    conds = [("independent", "independent"), ("correlated", "correlated\nshocks"), ("spillover", "spillover"),
             ("factor", "common\ndriver")]
    for k, (c, label) in enumerate(conds):
        rs = [r for r in e66["sim"] if r["cond"] == c]
        axes[0].plot(np.full(len(rs), k) + np.linspace(-0.12, 0.12, len(rs)), [r["S"] for r in rs], "o",
                     color=ORANGE if c == "spillover" else BLUE, ms=4)
        v = [x for r in rs for x in r["virtual"]]
        axes[0].plot(np.full(len(v), k + 0.3), v, "x", color="#999", ms=3)
    axes[0].set_xticks(range(len(conds)), [lab for _, lab in conds], fontsize=7.5)
    axes[0].set_ylabel("interaction share", fontsize=9)
    axes[0].set_title("Simulated markets:\nknown answers", fontsize=9.5, color=NAVY)
    ind = next(r for r in e66["real"] if r["name"] == "industries")
    ys = sorted(ind["years"], key=int)
    axes[1].bar([int(y) for y in ys], [ind["years"][y]["S"] for y in ys], color=BLUE, alpha=0.8)
    axes[1].set_ylim(-0.08, 0.08)
    axes[1].set_ylabel("interaction share (year)", fontsize=9)
    tw = axes[1].twinx()
    tw.plot([int(y) for y in ys], [ind["years"][y]["phi"] for y in ys], color=ORANGE, lw=1.5)
    tw.set_ylabel("turbulence (mean |return|, %)", fontsize=7.5, color=ORANGE)
    tw.tick_params(labelsize=8)
    axes[1].set_title(f"US market volatility, 49 industries:\nS = {ind['S']:+.3f} overall", fontsize=9.5,
                      color=NAVY)
    for ax in axes:
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig9_markets.svg")


def fig_memorygap():
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.e72_polarization_gap import table
    e67 = json.loads((ROOT / "results/e67_memory_gap_raw.json").read_text())
    fish = table(json.loads((ROOT / "results/e72_polarization_gap_raw.json").read_text()))
    e69 = json.loads((ROOT / "results/e69_market_log_volatility_raw.json").read_text())
    e70 = json.loads((ROOT / "results/e70_screen_raw.json").read_text())
    e71 = json.loads((ROOT / "results/e71_alm_holdout_raw.json").read_text())
    e73 = json.loads((ROOT / "results/e73_alm_replication_raw.json").read_text())
    e74 = json.loads((ROOT / "results/e74_nonlinear_gap_raw.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    sets = [("simulations", [(r["S_pred"], r["S"]) for r in e67["sim"]], GREY, "o"),
            ("221 brain areas", [(r["S_pred"], r["S"]) for r in e67["areas"]], NAVY, "o"),
            ("ALM, before the cue (59 sessions, 2 labs)", [(r["S_pred"], r["S_measured"]) for r in e70 if "S_measured" in r]
             + [(r["S_pred"], r["S"]) for r in e71["holdout"] + e73["replication"]], GREEN, "o"),
            ("markets: mean and breadth", [(r["log"]["S_pred"], r["log"]["S"]) for r in e69.values()]
             + [(r["breadth"]["S_pred"], r["breadth"]["S"]) for r in e74["real"]], ORANGE, "D"),
            ("fish, real (quadratic)", [(r[3]["S_q"], r[4]) for r in fish if r[2] == "real"], BLUE, "^"),
            ("fish, virtual", [(r[3]["S_q"], r[4]) for r in fish if r[2] == "virtual"], BLUE, "x")]
    for label, pts, c, m in sets:
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], m, color=c, ms=3.2 if m != "D" else 4.5, alpha=0.75, label=label,
                mfc="none" if m == "^" else c)
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
    for setter in (ax.set_xscale, ax.set_yscale):
        setter("symlog", linthresh=0.02)
    ax.set_xlim(-0.12, 1), ax.set_ylim(-0.12, 1)
    ticks = [0, 0.02, 0.1, 0.3, 1]
    ax.set_xticks([-0.1] + ticks, ["-0.1", "0", "0.02", "0.1", "0.3", "1"])
    ax.set_yticks([-0.1] + ticks, ["-0.1", "0", "0.02", "0.1", "0.3", "1"])
    ax.set_xlabel("predicted from two memories", fontsize=9)
    ax.set_ylabel("measured with fitted twins", fontsize=9)
    ax.set_title("One equation, every system", fontsize=9.5, color=NAVY)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left", handletextpad=0.2)
    ax = axes[1]
    doms = [("fish", BLUE), ("pigeons", GREEN), ("ALM", "#5aa36f"), ("markets", ORANGE)]
    for k, (d, c) in enumerate(doms):
        rs = [r for r in e70 if r["domain"] == d]
        jit = np.random.default_rng(k).uniform(-0.15, 0.15, len(rs))
        ax.plot(k + jit, [r["S_pred"] for r in rs], "o", color=c, ms=3.5, alpha=0.8)
        ax.plot(k + 0.32 + 0 * jit, [r["virtual"] for r in rs], "x", color="#999", ms=3.5)
    areas = [r["S_pred"] for r in e67["areas"]]
    ax.plot(4 + np.random.default_rng(9).uniform(-0.15, 0.15, len(areas)), areas, "o", color=NAVY, ms=2, alpha=0.5)
    ax.set_xticks(range(5), ["fish", "pigeons", "ALM", "markets", "brain\nareas"], fontsize=8)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(-0.002, 1)
    ax.set_yticks([0, 0.01, 0.03, 0.1, 0.3, 1], ["0", "0.01", "0.03", "0.1", "0.3", "1"])
    ax.set_ylabel("predicted interaction share", fontsize=9)
    ax.set_title("A screen of every dataset (x = virtual floor)", fontsize=9.5, color=NAVY)
    for a in axes:
        a.axhline(0, color="#bbb", lw=0.8)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig10_memorygap.svg")


def fig_extremes():
    load = lambda name: json.loads((ROOT / f"results/{name}_raw.json").read_text())  # noqa: E731
    e84, e85, e86 = load("e84_primes"), load("e85_prime_filters"), load("e86_ising")
    e87, e88, e89 = load("e87_randu"), load("e88_group_size"), load("e89_time_reversal")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    cols = [GREY, BLUE, ORANGE]

    ax = axes[0, 0]
    for (L, rows), c in zip(e86["scans"].items(), cols):
        ax.plot([r["T"] for r in rows], [r["S"] for r in rows], "o-", color=c, ms=2.5, lw=1.2, label=f"{L}×{L}")
    ax.axvline(2 / np.log(1 + np.sqrt(2)), color="black", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1.6, 2.269, 4, 10], ["1.6", "Tc", "4", "10"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("temperature", fontsize=9)
    ax.set_ylabel("interaction share", fontsize=9)
    ax.set_title("A. Ising magnet: peak at the transition", fontsize=9.5, color=NAVY)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[0, 1]
    for rule, c in zip(("mean-field", "topological", "metric"), (BLUE, ORANGE, NAVY)):
        rs = [r for r in e88 if r["rule"] == rule]
        ax.plot([r["N"] for r in rs], [r["S"] for r in rs], "o-", color=c, ms=3, lw=1.2, label=rule)
    rs = [r for r in e88 if r["rule"] == "mean-field"]
    ax.plot([r["N"] for r in rs], [r["theory"] for r in rs], color=GREY, lw=4, alpha=0.4, zorder=0,
            label="exact theory")
    ax.set_xscale("log")
    ax.set_xlabel("group size", fontsize=9)
    ax.set_title("B. Group size 2–10,000: saturation", fontsize=9.5, color=NAVY)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[0, 2]
    names = [k for k in e89 if k.startswith("leader")][:1] + [k for k in e89 if k.startswith("common")][:1] + \
        ["markets 1972-2026"]
    labels = ["leader", "common driver", "US markets"]
    x = np.arange(len(names))
    for dx, way, c in ((-0.2, "forward", BLUE), (0.2, "reversed", ORANGE)):
        ax.bar(x + dx, [e89[k][way]["S"] for k in names], 0.4, color=c, alpha=0.35)
        ax.plot(x + dx, [max(e89[k][way]["influence"]) for k in names], "D", color=c, ms=6, label=f"{way}")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("share (bars), largest influence (diamonds)", fontsize=8.5)
    ax.set_title("C. Time reversed: same share, leader gone", fontsize=9.5, color=NAVY)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1, 0]
    gens = ["pcg64", "minstd", "randu", "leaky"]
    for i, g in enumerate(gens):
        rs = [r for r in e87 if r["gen"] == g]
        ax.plot(i + np.linspace(-0.15, 0.15, len(rs)), [r["S"] for r in rs], "o", color=ORANGE if g == "randu"
                else BLUE if g == "leaky" else GREY, ms=4)
        caught = all(r["log10p"] < -10 for r in rs)
        ax.text(i, 0.0135, "3D test:\nrejects" if caught else "3D test:\npasses", ha="center", fontsize=7,
                color=ORANGE if caught else "#555")
    ax.axhline(np.mean([r["null_bar"] for r in e87]), color="black", lw=0.8, ls=":")
    ax.set_ylim(-0.003, 0.017)
    ax.set_xticks(range(4), ["PCG64", "MINSTD", "RANDU", "leaky"], fontsize=8)
    ax.set_ylabel("interaction share", fontsize=9)
    ax.set_title("D. Random numbers: two tools, two blind spots", fontsize=9.5, color=NAVY)

    ax = axes[1, 1]
    rng_labels = ["10^9 (dev.)", "10^12", "10^16"]
    runs = [e84["dev"]] + e84["tests"]
    keys = [("search_memory", "memory", BLUE), ("search_oracle", "oracle ceiling", GREY),
            ("deep_filter", "one more filter", ORANGE)]
    for j, (k, lab, c) in enumerate(keys):
        ax.bar(np.arange(3) + (j - 1) * 0.27, [100 * r["real"][k] for r in runs], 0.27, color=c, label=lab)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 1, 3, 10, 50], ["0", "1", "3", "10", "50"])
    ax.set_xticks(range(3), rng_labels, fontsize=8)
    ax.set_ylabel("more primes per test (%)", fontsize=9)
    ax.set_title("E. Primes: memory against divisibility", fontsize=9.5, color=NAVY)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1, 2]
    r = e85["tests"][-1]
    d = [int(k) for k in r["depths"]]
    m = list(r["depths"].values())
    ax.fill_between(d, [-300 * v["charged_sd"] for v in m], [300 * v["charged_sd"] for v in m], color="#ddd",
                    lw=0, label="chance")
    ax.plot(d, [100 * v["oracle"] for v in m], "s-", color=GREY, ms=3.5, label="oracle ceiling")
    ax.plot(d, [100 * v["free"] for v in m], "o-", color=BLUE, ms=3.5, label="memory, free")
    ax.plot(d, [100 * v["charged"] for v in m], "^--", color=ORANGE, ms=3.5, label="memory, charged")
    ax.set_xscale("log")
    ax.set_xlabel("trial division depth (numbers near 10^18)", fontsize=9)
    ax.set_ylabel("more primes per test (%)", fontsize=9)
    ax.set_title("F. All filters first: nothing left", fontsize=9.5, color=NAVY)
    ax.legend(fontsize=7.5, frameon=False)
    for a in axes.ravel():
        a.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig11_extremes.svg")


CSS = """
@page { size: A4; margin: 22mm 20mm 20mm; @bottom-center { content: counter(page);
        font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8.5pt; color: #7a8694; }
        @top-right { content: "Telling Coordination from Coincidence  \\00b7  E. J. Couture";
        font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8pt; color: #7a8694; } }
@page :first { margin: 0; @bottom-center { content: none; } @top-right { content: none; } }
body { font-family: "Charter", Georgia, serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a;
       margin: 0; }
h2, h3, th, figcaption, .cover, .box strong { font-family: "Helvetica Neue", Arial, sans-serif; }
h2 { font-size: 15pt; color: #1f3a5f; margin: 22pt 0 8pt; padding-bottom: 3pt;
     border-bottom: 1.2pt solid #1f3a5f; page-break-after: avoid; }
h2.newpage { page-break-before: always; }
.keep { page-break-inside: avoid; }
p { orphans: 3; widows: 3; }
h3 { font-size: 11.5pt; color: #1f3a5f; margin: 14pt 0 4pt; page-break-after: avoid; }
p { margin: 6pt 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 5pt 0; padding-left: 18pt; }
li { margin: 3pt 0; }
code { font-family: Menlo, monospace; font-size: 8.5pt; background: #f2f4f7; padding: 0 2pt; }
table { border-collapse: collapse; margin: 10pt 0; font-size: 9.2pt; width: 100%;
        page-break-inside: avoid; }
th { background: #1f3a5f; color: white; font-weight: 600; text-align: left; padding: 4pt 6pt; }
td { border-bottom: 0.5pt solid #c9d1db; padding: 4pt 6pt; vertical-align: top; }
tr:nth-child(even) td { background: #f5f7fa; }
figure { margin: 12pt 0; page-break-inside: avoid; text-align: center; }
figure img { width: 86%; }
figcaption { font-size: 8.8pt; color: #444; text-align: left; margin-top: 4pt; line-height: 1.4; }
.box { background: #eef3f9; border-left: 3pt solid #1f3a5f; padding: 6pt 12pt; margin: 12pt 0;
       page-break-inside: avoid; }
.box table { margin: 4pt 0 6pt; }
.cover { height: 297mm; box-sizing: border-box; padding: 0; position: relative;
         page-break-after: always; color: #1f3a5f; }
.cover .band { background: #1f3a5f; color: white; padding: 70mm 22mm 18mm; }
.cover .kicker { font-size: 10pt; letter-spacing: 2.5pt; text-transform: uppercase; opacity: 0.8; }
.cover h1 { font-size: 30pt; line-height: 1.15; margin: 10pt 0 12pt; font-weight: 700; }
.cover .sub { font-size: 14pt; line-height: 1.4; font-weight: 300; max-width: 150mm; }
.cover .meta { padding: 18mm 22mm; font-size: 12pt; }
.cover .author { font-size: 17pt; font-weight: 600; margin-bottom: 4pt; }
.cover .date { color: #7a8694; }
.cover .foot { position: absolute; bottom: 18mm; left: 22mm; right: 22mm; font-size: 8.5pt;
               color: #7a8694; border-top: 0.5pt solid #c9d1db; padding-top: 6pt; }
"""

COVER = f"""
<div class="cover">
  <div class="band">
    <div class="kicker">White paper</div>
    <h1>{TITLE}</h1>
    <div class="sub">{SUBTITLE}</div>
  </div>
  <div class="meta">
    <div class="author">{AUTHOR}</div>
    <div class="date">Independent researcher &nbsp;&middot;&nbsp; ORCID 0009-0004-3523-3469</div>
    <div class="date">{DATE} &nbsp;&middot;&nbsp; Version 2.3</div>
  </div>
  <div class="foot">&copy; 2026 {AUTHOR}. All results reported here were pre-registered;
  data sources are credited in the references.</div>
</div>
"""


def main():
    FIG.mkdir(exist_ok=True)
    fig_twin(), fig_walkers(), fig_transitions(), fig_fish(), fig_closure(), fig_influence(), fig_open(), fig_brain(), fig_markets()
    fig_memorygap(), fig_extremes()
    src = (HERE / "whitepaper.md").read_text()
    body = markdown.markdown(src, extensions=["tables", "sane_lists", "md_in_html"])
    for h in ("1. The problem", "7. Is the group", "8. A new domain", "9. A third domain", "10. One equation",
              "References"):
        body = body.replace(f"<h2>{h}", f"<h2 class=\"newpage\">{h}", 1)
    body = body.replace("<h2>15. Conclusion", "<div class=\"keep\"><h2>15. Conclusion", 1)
    body = body.replace("<h2>Appendix A", "</div><h2>Appendix A", 1)
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{TITLE}</title>"
            f"<style>{CSS}</style></head><body>{COVER}{body}</body></html>")
    html_path = HERE / "whitepaper.html"
    html_path.write_text(html)
    pdf = HERE / "Couture_Interaction_Share_White_Paper.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, capture_output=True)
    html_path.unlink()
    print(pdf)


if __name__ == "__main__":
    main()
