"""Experiment 8: negative control. Is flock "emergence" just persistence plus bookkeeping?

Psi = I(V_t; V_t+1) - sum_j I(X_j,t; V_t+1). If the parts X_j are independent,
the sum of their individual informations cannot exceed their joint
information, which equals I(V_t; V_t+1) when V_t+1 depends on the parts only
through V_t. So independent but individually persistent parts already give
Psi >= 0, with no interaction at all. Psi > 0 alone therefore does not show
that alignment creates a new layer.

Control: identical agents, speed, density, regions and estimators, but the
alignment radius is 0, so each agent ignores all others and its heading does
a random walk. Its noise is swept over a log grid and the control's best
value is used, giving it every chance. Flock numbers come from experiment 7.

Predictions, fixed before running:

  C1  The control's max Psi does not grow with N: for every (K, estimator),
      max Psi(N=3200) < 1.5 x max Psi(N=400).
  C2  At N = 3200, for every (K, estimator), every flock seed at the flock's
      peak noise exceeds every control seed at the control's peak noise.
  C3  For every (K, estimator), the flock's growth ratio
      max Psi(3200) / max Psi(400) exceeds the control's.

If C1 fails, the growth with N seen in experiments 6-7 is not evidence of
collective behaviour. If C2 and C3 pass, alignment adds emergence beyond
what independent persistent agents produce.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e4_flock_scaling import summarize
from experiments.e7_robustness import ESTIMATORS, GRIDS, SIZES, _key, simulate_psi

CONTROL_NOISES = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
SEEDS = 4


def _run(args):
    n, eta, seed = args
    return simulate_psi(n, eta, seed, radius=0.0)


def _peak(rows, key, n):
    s = summarize([r for r in rows if r["n"] == n], key=key)[n]
    eta = s["eta_psi_peak"]
    return s["psi_peak"], eta, [r[key] for r in rows if r["n"] == n and r["noise"] == eta]


def analyze(control: list[dict], flock: list[dict], out_dir: Path) -> str:
    lo, hi = min(SIZES), max(SIZES)
    lines = ["Experiment 8: non-interacting control vs flock (max Psi in bits)", ""]
    lines.append(f"  {'estimator':<10}{'K':>3}{'ctrl N=400':>12}{'ctrl N=3200':>13}{'ratio':>7}"
                 f"{'flock N=400':>13}{'flock N=3200':>14}{'ratio':>7}   seeds@3200 ctrl | flock")
    c1, c2, c3 = {}, {}, {}
    for tag, est in ESTIMATORS.items():
        for k in GRIDS:
            key = _key(tag, k)
            c_lo, _, _ = _peak(control, key, lo)
            c_hi, _, c_seeds = _peak(control, key, hi)
            f_lo, _, _ = _peak(flock, key, lo)
            f_hi, _, f_seeds = _peak(flock, key, hi)
            c_ratio = c_hi / c_lo if c_lo > 0 else np.inf
            f_ratio = f_hi / f_lo if f_lo > 0 else np.inf
            c1[key] = c_hi < 1.5 * c_lo
            c2[key] = min(f_seeds) > max(c_seeds)
            c3[key] = f_ratio > c_ratio
            lines.append(f"  {est:<10}{k:>3}{c_lo:12.3f}{c_hi:13.3f}{c_ratio:7.2f}{f_lo:13.3f}"
                         f"{f_hi:14.3f}{f_ratio:7.2f}   "
                         f"{max(c_seeds):+.2f} max | {min(f_seeds):+.2f} min")
    fails = lambda d: ", ".join(k for k, ok in d.items() if not ok) or "none"
    lines += ["", "  Pre-registered predictions:"]
    lines.append(f"  {'PASS' if all(c1.values()) else 'FAIL'}  C1 control Psi does not grow with N   "
                 f"[failing: {fails(c1)}]")
    lines.append(f"  {'PASS' if all(c2.values()) else 'FAIL'}  C2 flock beats control in every seed at N={hi}   "
                 f"[failing: {fails(c2)}]")
    lines.append(f"  {'PASS' if all(c3.values()) else 'FAIL'}  C3 flock grows faster with N than control   "
                 f"[failing: {fails(c3)}]")
    _plot(control, flock, out_dir / "e8_null_control.png")
    lines.append(f"\n  figure: {out_dir / 'e8_null_control.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(control, flock, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (tag, est) in zip(axes, ESTIMATORS.items()):
        for k, c in zip(GRIDS, ("tab:blue", "tab:orange", "tab:green")):
            key = _key(tag, k)
            ax.plot(SIZES, [_peak(flock, key, n)[0] for n in SIZES], "o-", color=c,
                    label=f"flock, {k} regions")
            ax.plot(SIZES, [_peak(control, key, n)[0] for n in SIZES], "s--", color=c, alpha=0.7,
                    label=f"no interaction, {k} regions")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES, [str(n) for n in SIZES])
        ax.set_yscale("symlog", linthresh=0.05)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("number of agents")
        ax.set_title(f"{est} estimator: best Psi over the noise sweep")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("max Psi (bits)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    flock_path = out_dir / "e7_robustness_raw.json"
    if not flock_path.exists():
        raise FileNotFoundError("run experiment 7 first: uv run python run.py robust")
    flock = json.loads(flock_path.read_text())
    tasks = [(n, eta, s) for n in sorted(SIZES, reverse=True) for eta in CONTROL_NOISES
             for s in range(2 if quick else SEEDS)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"  running {len(tasks)} control simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        control = list(pool.map(_run, tasks, chunksize=1))
    (out_dir / "e8_null_control_raw.json").write_text(json.dumps(control, indent=1))
    return analyze(control, flock, out_dir)


if __name__ == "__main__":
    main()
