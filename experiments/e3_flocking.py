"""Experiment 3: collective behaviour (Vicsek flocking).

Sweeps the noise level through the order/disorder transition and measures:

* Rosas et al.'s Psi for the flock's mean velocity V: does V predict its own
  future better than the individual agents? (Psi > 0 => causal emergence)
* the four-property report for the order parameter (polarization |V|) on a
  coarse time step, at a few representative noise levels.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from emergence_lab import (
    EmergenceReport,
    emergence_psi,
    macro_determinism,
    predictive_closure,
    print_reports,
    robustness,
)
from emergence_lab.systems.vicsek import Vicsek, headings, mean_velocity, polarization

MACRO_STEP = 10
"""Micro steps per macro step for the four-property report."""


def perturb(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(s), size=max(1, len(s) // 10), replace=False)
    s[idx, 2] = rng.uniform(-np.pi, np.pi, len(idx))
    return s


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    base = Vicsek(n_agents=60 if quick else 100)
    n_steps, burn = (2000, 500) if quick else (5000, 1000)
    n_seeds = 1 if quick else 3
    noises = np.linspace(0.5, 6.0, 8 if quick else 12)
    report_at = {0, len(noises) // 2, len(noises) - 1}

    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"== Psi for mean velocity V across the flocking transition "
        f"(N={base.n_agents}, {n_steps} samples, {n_seeds} seed(s)) ==")
    say(f"  {'noise':>6}{'|V|':>8}{'Psi':>10}{'Psi_corr':>16}{'I(V;V+)':>10}{'max I(Xj;V+)':>14}")
    sweep, reports = [], []
    for k, eta in enumerate(noises):
        model = replace(base, noise=float(eta))
        runs = [model.simulate(n_steps, burn, seed=100 * k + s) for s in range(n_seeds)]
        psis = []
        for traj in runs:
            V = mean_velocity(traj)
            psis.append(emergence_psi(V[:-1], V[1:], headings(traj[:-1])))
        pc_mean = np.mean([p["psi_corrected"] for p in psis])
        pc_sd = np.std([p["psi_corrected"] for p in psis])
        pol = np.mean([polarization(t).mean() for t in runs])
        sweep.append((eta, pol, pc_mean, pc_sd))
        say(f"  {eta:6.2f}{pol:8.3f}{np.mean([p['psi'] for p in psis]):10.2f}"
            f"{pc_mean:9.2f} ±{pc_sd:5.2f}{np.mean([p['I(V_t;V_t+1)'] for p in psis]):10.3f}"
            f"{np.mean([p['max_I(X_j;V_t+1)'] for p in psis]):14.3f}")

        if k in report_at:
            reports.append(_four_properties(model, runs[0], eta, seed=k))

    say(f"\n== Four-property report: polarization |V| over {MACRO_STEP}-step macro time ==")
    lines.append(print_reports(reports))
    for r in reports:
        say(f"    {r.macro:<34} R2 macro-only={r.extra['r2_macro']:.3f}  "
            f"R2 with micro={r.extra['r2_full']:.3f}")

    if out_dir is not None:
        _plot(np.array(sweep), out_dir / "e3_flocking.png")
        say(f"\n  figure: {out_dir / 'e3_flocking.png'}")
    return "\n".join(lines)


def _four_properties(model: Vicsek, traj: np.ndarray, eta: float, seed: int) -> EmergenceReport:
    coarse = traj[::MACRO_STEP]
    p = polarization(coarse)[:, None]
    micro = headings(coarse[:-1]).reshape(len(coarse) - 1, -1)
    blocks = np.arange(len(coarse) - 1) // 25
    pc = predictive_closure(p[:-1], p[1:], micro, groups=blocks)
    rng = np.random.default_rng(seed)
    starts = [traj[i] for i in rng.choice(len(traj), 20, replace=False)]

    def macro_step(s, r):
        for _ in range(MACRO_STEP):
            s = model.step(s, r)
        return s

    return EmergenceReport(
        system="vicsek",
        macro=f"polarization, noise={eta:.2f}",
        compression=1 - 1 / (3 * model.n_agents),
        autonomy=pc["autonomy"],
        determinism=macro_determinism(p[:-1], p[1:], groups=blocks),
        robustness=robustness(starts, macro_step, polarization, perturb, 5),
        extra=pc,
    )


def _plot(sweep: np.ndarray, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(sweep[:, 0], sweep[:, 1], "o-", color="tab:blue")
    ax1.set_xlabel("noise")
    ax1.set_ylabel("polarization |V|", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.errorbar(sweep[:, 0], sweep[:, 2], yerr=sweep[:, 3], fmt="s-", color="tab:red", capsize=3)
    ax2.axhline(0, color="tab:red", lw=0.8, ls="--")
    ax2.set_ylabel("redundancy-corrected Psi (bits)", color="tab:red")
    ax1.set_title("Vicsek flock: order and causal emergence vs noise")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
