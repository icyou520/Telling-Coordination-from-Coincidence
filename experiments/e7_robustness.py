"""Experiment 7: is the large-flock causal-emergence result robust?

Experiment 6 found that Psi of the flock's mean velocity, measured against 8
spatial regions with a Gaussian estimator, becomes positive and grows with
flock size, peaking just on the disordered side of the transition. This
experiment varies the analysis choices that could have produced that:

  * number of regions K: 4 (2x2), 8 (4x2), 16 (4x4)
  * information estimator: Gaussian, and the nonparametric KSG estimator
  * flock size extended to N = 3200

Predictions, fixed before running:

  R1  For every (K, estimator), max Psi over the sweep increases with N at
      every step 400 -> 800 -> 1600 -> 3200.
  R2  For every (K, estimator), at N = 3200 Psi at its peak noise is positive
      in every seed.
  R3  For K = 8 with the Gaussian estimator at N = 3200, the Psi peak lies on
      the disordered side of the susceptibility peak and within 0.75 of it:
      0 <= eta_Psi - eta_chi <= 0.75.

The result counts as robust only if R2 passes for all six combinations.

Estimator calibration, checked on synthetic Gaussian data (n = 6000) before
running: KSG is accurate up to ~6 bits but underestimates strong dependence
(8.14 vs 8.97 true bits at rho = 0.999). The macro term I(V_t; V_t+1) is in
that range, so KSG Psi is biased downward: a conservative check.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e4_flock_scaling import summarize
from experiments.e5_flock_regions import region_parts

SIZES = (400, 800, 1600, 3200)
NOISES = np.round(np.arange(2.5, 5.51, 0.25), 3)
SEEDS = 4
N_STEPS, BURN_IN = 6000, 4000
GRIDS = {4: (2, 2), 8: (4, 2), 16: (4, 4)}
ESTIMATORS = {"g": "gaussian", "k": "ksg"}


def _key(est: str, k: int) -> str:
    return f"psi_{est}{k}"


def _run(args):
    n, eta, seed = args
    return simulate_psi(n, eta, seed)


def simulate_psi(n: int, eta: float, seed: int, radius: float = 1.0) -> dict:
    """One flock run: Psi of the mean velocity against 4, 8 and 16 regions, both estimators.

    ``radius = 0`` switches alignment off: each agent only sees itself.
    """
    from emergence_lab.info import emergence_psi
    from emergence_lab.systems.vicsek import Vicsek

    model = Vicsek(n_agents=n, noise=float(eta), radius=radius)
    rng = np.random.default_rng(20_000 * n + 101 * seed + int(eta * 1000))
    s = model.initial(rng)
    for _ in range(BURN_IN):
        s = model.step(s, rng)
    parts = {k: np.empty((N_STEPS, k, 2)) for k in GRIDS}
    chunk = np.empty((500, n, 3))
    for start in range(0, N_STEPS, len(chunk)):
        for i in range(len(chunk)):
            s = model.step(s, rng)
            chunk[i] = s
        for k, grid in GRIDS.items():
            parts[k][start : start + len(chunk)] = region_parts(chunk, model.box, grid)
    V = parts[4].mean(1)
    pol = np.linalg.norm(V, axis=1)
    row = {"n": n, "noise": float(eta), "seed": seed,
           "pol_mean": float(pol.mean()), "chi": float(n * pol.var())}
    for k in GRIDS:
        for tag, est in ESTIMATORS.items():
            row[_key(tag, k)] = emergence_psi(V[:-1], V[1:], parts[k][:-1], estimator=est)["psi"]
    return row


def analyze(rows: list[dict], out_dir: Path) -> str:
    sizes = sorted({r["n"] for r in rows})
    big = sizes[-1]
    lines = ["Experiment 7: robustness of large-flock causal emergence", ""]
    lines.append(f"  {'estimator':<10}{'K':>3}" + "".join(f"{'N=' + str(n):>10}" for n in sizes)
                 + f"{'peak@' + str(big):>10}{'chi@' + str(big):>9}  seeds at peak (N={big})")
    r1, r2, summaries = {}, {}, {}
    for tag, est in ESTIMATORS.items():
        for k in GRIDS:
            key = _key(tag, k)
            s = summarize(rows, key=key)
            summaries[key] = s
            peaks = [s[n]["psi_peak"] for n in sizes]
            eta = s[big]["eta_psi_peak"]
            seeds = [r[key] for r in rows if r["n"] == big and r["noise"] == eta]
            r1[key] = all(b > a for a, b in zip(peaks, peaks[1:]))
            r2[key] = all(v > 0 for v in seeds)
            lines.append(f"  {est:<10}{k:>3}" + "".join(f"{p:10.3f}" for p in peaks)
                         + f"{eta:10.2f}{s[big]['eta_chi_peak']:9.2f}  "
                         + ", ".join(f"{v:+.2f}" for v in seeds))
    s8 = summaries[_key("g", 8)][big]
    offset = s8["eta_psi_peak"] - s8["eta_chi_peak"]
    lines += ["", "  (table: max Psi in bits over the noise sweep, mean of seeds)", "",
              "  Pre-registered predictions:"]
    fails = lambda d: ", ".join(k for k, ok in d.items() if not ok) or "none"
    lines.append(f"  {'PASS' if all(r1.values()) else 'FAIL'}  R1 max Psi increases with N "
                 f"for every (K, estimator)   [failing: {fails(r1)}]")
    lines.append(f"  {'PASS' if all(r2.values()) else 'FAIL'}  R2 Psi > 0 in every seed at N={big} "
                 f"for every (K, estimator)   [failing: {fails(r2)}]")
    lines.append(f"  {'PASS' if 0 <= offset <= 0.75 + 1e-9 else 'FAIL'}  R3 K=8 Gaussian peak is "
                 f"0..0.75 above the chi peak at N={big}   [offset = {offset:+.2f}]")
    lines.append(f"\n  VERDICT: {'ROBUST' if all(r2.values()) else 'NOT ROBUST'} (criterion: R2 for all six)")
    _plot(summaries, sizes, out_dir / "e7_robustness.png")
    lines.append(f"  figure: {out_dir / 'e7_robustness.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(summaries: dict, sizes: list[int], path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(sizes)))
    for row, (tag, est) in enumerate(ESTIMATORS.items()):
        for col, k in enumerate(GRIDS):
            ax = axes[row, col]
            s = summaries[_key(tag, k)]
            for n, c in zip(sizes, colors):
                x = np.array(s[n]["noises"])
                ax.errorbar(x, s[n]["psi_corrected_mean"], yerr=s[n]["psi_corrected_sd"], fmt="o-",
                            ms=3, color=c, capsize=2, label=f"N={n}")
                ax.axvline(s[n]["eta_chi_peak"], color=c, ls=":", lw=1)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_yscale("symlog", linthresh=0.5)
            ax.set_title(f"{est} estimator, {k} regions", fontsize=10)
            ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xlabel("noise")
            if col == 0:
                ax.set_ylabel("Psi (bits)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Causal emergence of flock velocity vs regional parts (dotted = susceptibility peak)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = (400, 800) if quick else SIZES
    seeds = 2 if quick else SEEDS
    tasks = [(n, eta, s) for n in sorted(sizes, reverse=True) for eta in NOISES for s in range(seeds)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"  running {len(tasks)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(_run, tasks, chunksize=1))
    (out_dir / "e7_robustness_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows, out_dir)


if __name__ == "__main__":
    main()
