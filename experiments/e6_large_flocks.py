"""Experiment 6: does flock-level causal emergence appear near the transition in large flocks?

Experiment 5 (Psi of the mean velocity against 8 fixed spatial regions) failed
its predictions, but showed an unplanned trend: the maximum of Psi rose with
flock size (-0.015, -0.013, +0.002, +0.110 for N = 50..400) and moved toward
the susceptibility peak. That trend was noticed after the fact, so it is
tested here as a new hypothesis on larger flocks, a finer noise grid around
the transition, and longer burn-in.

Predictions, fixed before running (plain Psi vs 8 regions):

  Q1  The maximum of Psi over the sweep increases with N at every step
      200 -> 400 -> 800 -> 1600.
  Q2  At N = 1600, Psi at its peak noise is positive in every seed.
  Q3  The gap |eta_Psi - eta_chi| at N = 1600 is smaller than at N = 400
      and at most one grid step (0.25).

If Q1 fails, the experiment-5 trend was noise. If Q1 holds but Q3 fails,
large flocks show emergence, but not specifically at the transition.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e4_flock_scaling import plot, summarize
from experiments.e5_flock_regions import region_parts

SIZES = (200, 400, 800, 1600)
NOISES = np.round(np.arange(2.5, 5.51, 0.25), 3)
SEEDS = 4
N_STEPS, BURN_IN = 6000, 4000
GRID_STEP = 0.25


def _run(args):
    n, eta, seed = args
    from emergence_lab.info import emergence_psi
    from emergence_lab.systems.vicsek import Vicsek

    model = Vicsek(n_agents=n, noise=float(eta))
    rng = np.random.default_rng(10_000 * n + 101 * seed + int(eta * 1000))
    s = model.initial(rng)
    for _ in range(BURN_IN):
        s = model.step(s, rng)
    V = np.empty((N_STEPS, 2))
    parts = np.empty((N_STEPS, 8, 2))
    chunk = np.empty((500, n, 3))
    for start in range(0, N_STEPS, len(chunk)):
        for k in range(len(chunk)):
            s = model.step(s, rng)
            chunk[k] = s
        sl = slice(start, start + len(chunk))
        parts[sl] = region_parts(chunk, model.box)
        V[sl] = parts[sl].mean(1)
    psi = emergence_psi(V[:-1], V[1:], parts[:-1])
    pol = np.linalg.norm(V, axis=1)
    return {
        "n": n, "noise": float(eta), "seed": seed,
        "psi": psi["psi"], "self_info": psi["I(V_t;V_t+1)"],
        "sum_parts": psi["sum_I(X_j;V_t+1)"], "max_part": psi["max_I(X_j;V_t+1)"],
        "pol_mean": float(pol.mean()), "chi": float(n * pol.var()),
    }


def verdicts(summary: dict, rows: list[dict]) -> list[str]:
    sizes = sorted(summary)
    peaks = [summary[n]["psi_peak"] for n in sizes]
    gap = lambda n: abs(summary[n]["eta_psi_peak"] - summary[n]["eta_chi_peak"])
    big = sizes[-1]
    eta_big = summary[big]["eta_psi_peak"]
    seed_vals = [r["psi"] for r in rows if r["n"] == big and r["noise"] == eta_big]
    checks = [
        ("Q1 max Psi increases with N at every step", all(b > a for a, b in zip(peaks, peaks[1:]))),
        (f"Q2 Psi > 0 in every seed at N={big}'s peak noise", all(v > 0 for v in seed_vals)),
        (f"Q3 peak gap at N={big} < gap at N=400 and <= {GRID_STEP}",
         gap(big) < gap(400) and gap(big) <= GRID_STEP + 1e-9),
    ]
    out = [f"  {'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks]
    out.append(f"        (Psi per seed at N={big}, noise {eta_big}: "
               + ", ".join(f"{v:+.3f}" for v in seed_vals) + ")")
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = (200, 400) if quick else SIZES
    seeds = 2 if quick else SEEDS
    tasks = [(n, eta, s) for n in sorted(sizes, reverse=True) for eta in NOISES for s in range(seeds)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"  running {len(tasks)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(_run, tasks, chunksize=1))
    (out_dir / "e6_large_flocks_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows, out_dir)


def analyze(rows: list[dict], out_dir: Path) -> str:
    summary = summarize(rows, key="psi")
    lines = ["Experiment 6: Psi vs 8 regions in large flocks", ""]
    lines.append(f"  {'N':>5}{'chi peak':>10}{'Psi peak':>10}{'gap':>7}{'max Psi':>10}{'±sd':>8}")
    for n in sorted(summary):
        s = summary[n]
        i = s["noises"].index(s["eta_psi_peak"])
        lines.append(f"  {n:5d}{s['eta_chi_peak']:10.3f}{s['eta_psi_peak']:10.3f}"
                     f"{abs(s['eta_psi_peak'] - s['eta_chi_peak']):7.3f}{s['psi_peak']:10.3f}"
                     f"{s['psi_corrected_sd'][i]:8.3f}")
    lines += ["", "  Pre-registered predictions:"] + verdicts(summary, rows)
    plot(summary, out_dir / "e6_large_flocks.png", psi_title="Psi vs 8 regions (dotted = chi peak)")
    lines.append(f"\n  figure: {out_dir / 'e6_large_flocks.png'}")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
