"""Experiment 4: does causal emergence in a flock peak at the order/disorder transition?

Experiment 3 found that the redundancy-corrected Psi of the flock's mean
velocity is negative in the ordered phase, positive through the transition and
~0 in the disordered phase, for N = 100 agents. This experiment checks whether
that is a real effect or a small-system artifact by repeating it for several
flock sizes and comparing against an independent locator of the transition:
the susceptibility chi = N * Var(polarization), whose peak marks the critical
noise for a finite flock.

Predictions, fixed before running:

  P1  For every N, corrected Psi is positive somewhere in the sweep.
  P2  The noise at which corrected Psi peaks lies within one grid step
      (0.375) of the susceptibility peak, for every N.
  P3  As N grows, the gap |eta_Psi - eta_chi| does not grow.
  P4  Deep in the ordered phase (noise <= 1.5) corrected Psi is negative
      for every N.

If P1 and P2 hold, "emergence peaks at criticality" survives this test.
If they fail for the larger flocks, the N = 100 result was an artifact.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIZES = (50, 100, 200, 400)
NOISES = np.round(np.arange(1.0, 5.51, 0.375), 3)
SEEDS = 4
N_STEPS, BURN_IN = 6000, 2000


def _run(args):
    n, eta, seed = args
    from emergence_lab.info import emergence_psi
    from emergence_lab.systems.vicsek import Vicsek, headings, mean_velocity, polarization

    traj = Vicsek(n_agents=n, noise=float(eta)).simulate(N_STEPS, BURN_IN, seed=1000 * n + 17 * seed + int(eta * 1000))
    V = mean_velocity(traj)
    psi = emergence_psi(V[:-1], V[1:], headings(traj[:-1]))
    pol = polarization(traj)
    return {
        "n": n, "noise": float(eta), "seed": seed,
        "psi": psi["psi"], "psi_corrected": psi["psi_corrected"],
        "self_info": psi["I(V_t;V_t+1)"], "max_part": psi["max_I(X_j;V_t+1)"],
        "pol_mean": float(pol.mean()), "chi": float(n * pol.var()),
    }


def run_sweep(sizes=SIZES, noises=NOISES, seeds=SEEDS, workers: int | None = None) -> list[dict]:
    tasks = [(n, eta, s) for n in sorted(sizes, reverse=True) for eta in noises for s in range(seeds)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        return list(pool.map(_run, tasks, chunksize=1))


def summarize(rows: list[dict], key: str = "psi_corrected") -> dict:
    out = {}
    for n in sorted({r["n"] for r in rows}):
        sub = [r for r in rows if r["n"] == n]
        noises = sorted({r["noise"] for r in sub})
        stat = lambda k: np.array([[r[k] for r in sub if r["noise"] == e] for e in noises])
        psi_c, chi, pol = stat(key), stat("chi"), stat("pol_mean")
        psi_mean = psi_c.mean(1)
        chi_mean = chi.mean(1)
        out[n] = {
            "noises": noises,
            "psi_corrected_mean": psi_mean.tolist(),
            "psi_corrected_sd": psi_c.std(1).tolist(),
            "chi_mean": chi_mean.tolist(),
            "chi_sd": chi.std(1).tolist(),
            "pol_mean": pol.mean(1).tolist(),
            "eta_psi_peak": noises[int(np.argmax(psi_mean))],
            "eta_chi_peak": noises[int(np.argmax(chi_mean))],
            "psi_peak": float(psi_mean.max()),
            "psi_ordered_max": float(max((m for e, m in zip(noises, psi_mean) if e <= 1.5),
                                         default=np.nan)),
        }
    return out


def verdicts(summary: dict, grid_step: float = 0.375, label: str = "corrected Psi") -> list[str]:
    sizes = sorted(summary)
    gaps = [abs(summary[n]["eta_psi_peak"] - summary[n]["eta_chi_peak"]) for n in sizes]
    checks = [
        (f"P1 {label} > 0 somewhere, every N", all(summary[n]["psi_peak"] > 0 for n in sizes)),
        (f"P2 {label} peak within one grid step of chi peak, every N",
         all(g <= grid_step + 1e-9 for g in gaps)),
        ("P3 peak gap does not grow with N", gaps[-1] <= gaps[0] + 1e-9),
        (f"P4 {label} < 0 in ordered phase (noise <= 1.5), every N",
         all(summary[n]["psi_ordered_max"] < 0 for n in sizes)),
    ]
    return [f"  {'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks]


def plot(summary: dict, path: Path, psi_title: str = "Causal emergence: corrected Psi (dotted = chi peak)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(summary)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(sizes)))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for n, c in zip(sizes, colors):
        s = summary[n]
        x = np.array(s["noises"])
        ax[0].plot(x, s["pol_mean"], "o-", color=c, label=f"N={n}")
        ax[1].errorbar(x, s["chi_mean"], yerr=s["chi_sd"], fmt="o-", color=c, capsize=2, label=f"N={n}")
        ax[2].errorbar(x, s["psi_corrected_mean"], yerr=s["psi_corrected_sd"], fmt="s-", color=c,
                       capsize=2, label=f"N={n}")
        ax[1].axvline(s["eta_chi_peak"], color=c, ls=":", lw=1)
        ax[2].axvline(s["eta_chi_peak"], color=c, ls=":", lw=1)
    ax[0].set_title("Order: polarization")
    ax[1].set_title("Transition locator: susceptibility N·Var(|V|)")
    ax[2].set_title(psi_title)
    ax[2].axhline(0, color="k", lw=0.7)
    ax[2].set_yscale("symlog", linthresh=1.0)
    for a in ax:
        a.set_xlabel("noise")
        a.legend(fontsize=8)
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = (50, 100) if quick else SIZES
    seeds = 2 if quick else SEEDS
    print(f"  running {len(sizes) * len(NOISES) * seeds} simulations ...", flush=True)
    rows = run_sweep(sizes, NOISES, seeds)
    (out_dir / "e4_flock_scaling_raw.json").write_text(json.dumps(rows, indent=1))
    summary = summarize(rows)

    lines = [__doc__.split("Predictions")[0].strip().splitlines()[0], ""]
    lines.append(f"  {'N':>5}{'chi peak':>10}{'Psi peak':>10}{'gap':>7}{'max Psi_corr':>14}"
                 f"{'Psi_corr @ noise<=1.5':>23}")
    for n in sorted(summary):
        s = summary[n]
        lines.append(f"  {n:5d}{s['eta_chi_peak']:10.3f}{s['eta_psi_peak']:10.3f}"
                     f"{abs(s['eta_psi_peak'] - s['eta_chi_peak']):7.3f}{s['psi_peak']:14.2f}"
                     f"{s['psi_ordered_max']:23.2f}")
    lines += ["", "  Pre-registered predictions:"] + verdicts(summary)
    plot(summary, out_dir / "e4_flock_scaling.png")
    lines.append(f"\n  figure: {out_dir / 'e4_flock_scaling.png'}")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
