"""Exploratory diagnostic after e43 (not pre-registered).

Why does stage 1 eliminate "none" on virtual groups (no interaction), while
attraction/cohesion rules survive? Hypothesis: compensation. The no-interaction
model's 1-3 s forecasts of the macro variables are biased (e.g. simulated
fish drift apart faster than real ones), and a fitted attraction term pulls
the forecast back toward the data without any real interaction.

For each virtual 8-fish group: signed forecast error (prediction - observed)
of every macro variable at 1 s and 3 s for "none" and for k6-C, on the same
150 states as e43's stage 1, plus the fitted k6-C social coefficients.

    uv run python -m experiments.e43b_virtual_diagnostic
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import TRAIN, kinematics, virtual
from experiments.e39_multiscale_elimination import macro
from experiments.e43_elimination_fish_maps import HORIZONS, N_STATES, REPLICAS, fit_all, model_step
from experiments.fish_data import load

NAMES = ("polarization", "milling", "spread", "NND")
CANDS = ("none", "k6-C")


def _one(args):
    n, rec, kind = args
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    sp = np.where(sp > 100.0, 0.0, sp)
    T = len(th)
    cut = int(TRAIN * T)
    fits, _, _, _, (l1, l2, _, dvp) = fit_all(p, th, sp, cut, CANDS)
    idx = np.linspace(cut, T - max(HORIZONS) - 1, N_STATES).astype(int)
    obs = {s: macro(p[idx + s], th[idx + s]) for s in HORIZONS}
    out = {"N": n, "rec": rec, "kind": kind, "coef_k6C_turn": fits["k6-C"][1].tolist(),
           "coef_k6C_speed": fits["k6-C"][4].tolist()}
    for h in CANDS:
        rng = np.random.default_rng(7)
        st = tuple(np.repeat(a[idx][:, None], REPLICAS, 1) for a in (p, th, sp, l1, l2, dvp))
        for s in range(1, max(HORIZONS) + 1):
            st = model_step(st, fits[h], h, rng)
            if s in HORIZONS:
                d = macro(st[0], st[1]).mean(1) - obs[s]
                out[f"{h}@{s}"] = {"bias": d.mean(0).tolist(), "rmse": np.sqrt((d ** 2).mean(0)).tolist(),
                                   "se": (d.std(0, ddof=1) / np.sqrt(len(d))).tolist()}
    return out


def main():
    tasks = [(8, r, k) for r in range(6) for k in ("virtual", "real")]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(_one, tasks))
    Path("results/e43b_virtual_diagnostic_raw.json").write_text(json.dumps(rows))
    lines = ["e43b (exploratory): stage-1 forecast bias, prediction - observed, 8-fish groups", ""]
    for s in HORIZONS:
        lines.append(f"  horizon {s / 10:.0f} s:  bias (rmse) for " + ", ".join(NAMES))
        for r in rows:
            for h in CANDS:
                st = r[f"{h}@{s}"]
                lines.append(f"    {r['kind']:7} #{r['rec']} {h:5} " + "  ".join(
                    f"{b:+.3f} ({m:.3f})" for b, m in zip(st["bias"], st["rmse"])))
        lines.append("")
    lines.append("  k6-C coefficients: turn (attraction, cohesion) | speed (ahead, cohesion)")
    for r in rows:
        lines.append(f"    {r['kind']:7} #{r['rec']}  " + " ".join(f"{x:+.4f}" for x in r["coef_k6C_turn"])
                     + " | " + " ".join(f"{x:+.4f}" for x in r["coef_k6C_speed"]))
    text = "\n".join(lines)
    print(text)
    Path("results/e43b_virtual_diagnostic.txt").write_text(text)


if __name__ == "__main__":
    main()
