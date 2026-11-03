"""Experiment 5: causal emergence of a flock measured against a fixed number of parts.

Experiment 4 showed that Psi computed against individual agents is dominated
by flock size: the sum over parts grows with N. Here the parts are instead 8
spatial regions (a 4 x 2 grid over the box), whatever the flock size. Part j
is the summed heading of the agents in region j divided by N/8, so the flock's
mean velocity V is exactly the average of the 8 parts.

Primary measure: plain Psi (a rigorous sufficient condition for emergence).
Secondary, exploratory: redundancy-corrected Psi.

Predictions, fixed before running (same as experiment 4, for plain Psi):

  P1  For every N, Psi is positive somewhere in the sweep.
  P2  The noise at which Psi peaks lies within one grid step (0.375) of the
      susceptibility peak, for every N.
  P3  As N grows, the gap |eta_Psi - eta_chi| does not grow.
  P4  Deep in the ordered phase (noise <= 1.5) Psi is negative for every N.

Same simulations as experiment 4 (same seeds), so the susceptibility peaks
are directly comparable.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e4_flock_scaling import BURN_IN, N_STEPS, NOISES, SEEDS, SIZES, plot, summarize, verdicts

GRID = (4, 2)


def region_parts(traj: np.ndarray, box: float, grid: tuple[int, int] = GRID) -> np.ndarray:
    """(T, N, 3) states -> (T, gx*gy, 2) regional heading sums, scaled so their mean is V."""
    gx, gy = grid
    k = gx * gy
    n = traj.shape[1]
    ix = np.floor(traj[..., 0] / box * gx).astype(int) % gx
    iy = np.floor(traj[..., 1] / box * gy).astype(int) % gy
    region = iy * gx + ix
    T = len(traj)
    flat = (np.arange(T)[:, None] * k + region).ravel()
    parts = np.empty((T, k, 2))
    for c, w in enumerate((np.cos(traj[..., 2]), np.sin(traj[..., 2]))):
        parts[:, :, c] = np.bincount(flat, weights=w.ravel(), minlength=T * k).reshape(T, k)
    return parts / (n / k)


def _run(args):
    n, eta, seed = args
    from emergence_lab.info import emergence_psi
    from emergence_lab.systems.vicsek import Vicsek, mean_velocity, polarization

    model = Vicsek(n_agents=n, noise=float(eta))
    traj = model.simulate(N_STEPS, BURN_IN, seed=1000 * n + 17 * seed + int(eta * 1000))
    V = mean_velocity(traj)
    parts = region_parts(traj, model.box)
    assert np.allclose(parts.mean(1), V)
    psi = emergence_psi(V[:-1], V[1:], parts[:-1])
    pol = polarization(traj)
    return {
        "n": n, "noise": float(eta), "seed": seed,
        "psi": psi["psi"], "psi_corrected": psi["psi_corrected"],
        "self_info": psi["I(V_t;V_t+1)"], "sum_parts": psi["sum_I(X_j;V_t+1)"],
        "max_part": psi["max_I(X_j;V_t+1)"],
        "pol_mean": float(pol.mean()), "chi": float(n * pol.var()),
    }


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = (50, 100) if quick else SIZES
    seeds = 2 if quick else SEEDS
    tasks = [(n, eta, s) for n in sorted(sizes, reverse=True) for eta in NOISES for s in range(seeds)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"  running {len(tasks)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(_run, tasks, chunksize=1))
    (out_dir / "e5_flock_regions_raw.json").write_text(json.dumps(rows, indent=1))

    lines = ["Experiment 5: Psi of the flock against 8 fixed spatial regions", ""]
    for key, label, tag in (("psi", "Psi", "PRIMARY"), ("psi_corrected", "corrected Psi", "EXPLORATORY")):
        summary = summarize(rows, key=key)
        lines.append(f"  [{tag}] {label}")
        lines.append(f"  {'N':>5}{'chi peak':>10}{'Psi peak':>10}{'gap':>7}{'max':>9}{'max @ noise<=1.5':>18}")
        for n in sorted(summary):
            s = summary[n]
            lines.append(f"  {n:5d}{s['eta_chi_peak']:10.3f}{s['eta_psi_peak']:10.3f}"
                         f"{abs(s['eta_psi_peak'] - s['eta_chi_peak']):7.3f}{s['psi_peak']:9.3f}"
                         f"{s['psi_ordered_max']:18.3f}")
        lines += verdicts(summary, label=label) + [""]
        plot(summary, out_dir / f"e5_flock_regions_{key}.png",
             psi_title=f"{label} vs 8 regions (dotted = chi peak)")
    lines.append(f"  figures: {out_dir / 'e5_flock_regions_psi.png'}, "
                 f"{out_dir / 'e5_flock_regions_psi_corrected.png'}")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
