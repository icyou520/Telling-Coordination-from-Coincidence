"""Experiment 22: fresh-size confirmation of the calibrated field law.

Experiment 21 calibrated one offset (DELTA = +0.05) at N = 1600 and predicted
mean global order at N = 400 and 6400. Here the law is frozen exactly as
there (fitted at N = 1600, DELTA = +0.05) and tested on a size used nowhere
before, N = 3600 (box 30, a 6 x 6 lattice of cells of side 5), with new seeds.

Baselines:
  no size dependence  the N = 1600 agent curve;
  interpolation       agent curves at N = 1600 and 6400 interpolated linearly
                      in log N (uses real data the field law never saw).

Predictions, fixed before running:

  F1  Mean global phi within 0.05 of the agents at >= 10/13 noise levels.
  F2  At noise 3.0, predicted phi within 0.05 of the agents.
  F3  Smaller summed |error| than the no-size-dependence baseline.
  F4  Summed |error| no more than 1.5x that of the interpolation baseline.
  F5  Susceptibility peak within 0.25 of the agents' peak.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e9_interaction_closure import BURN_IN, N_MACRO, TAU
from experiments.e11_band_closure import NOISES
from experiments.e20_big_cell_field import N_FIT, simulate_fields
from experiments.e21_calibrated_field import _law

N_TEST = 3600
G_TEST = 6
DELTA = 0.05
SEEDS = range(1000, 1004)


def _agent_run(args):
    eta, seed = args
    from emergence_lab.systems.vicsek import Vicsek

    model = Vicsek(n_agents=N_TEST, noise=float(eta))
    rng = np.random.default_rng(110_000 + 101 * seed + int(eta * 1000))
    s = model.initial(rng)
    for _ in range(BURN_IN):
        s = model.step(s, rng)
    phi = np.empty(N_MACRO)
    for t in range(N_MACRO):
        for _ in range(TAU):
            s = model.step(s, rng)
        phi[t] = np.hypot(np.cos(s[:, 2]).mean(), np.sin(s[:, 2]).mean())
    return float(eta), float(phi.mean()), float(N_TEST * phi.var())


def _field_run(args):
    eta, fields = args
    phi = _law(eta, fields, DELTA).simulate(G_TEST, seed=int(eta * 100) + 2)
    return float(eta), float(phi.mean()), float(N_TEST * phi.var())


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from experiments.e17_field_law import _agents

    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    workers = max(1, (os.cpu_count() or 2) - 2)
    d = np.load(simulate_fields(out_dir, quick))
    jobs = [(float(e), d["fields"][d["noise"] == e]) for e in NOISES]
    seeds = list(SEEDS)[:1] if quick else list(SEEDS)
    print(f"  simulating {len(NOISES) * len(seeds)} agent runs at N={N_TEST} "
          f"and the frozen field law ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        field = {e: (p, c) for e, p, c in pool.map(_field_run, jobs)}
        runs = list(pool.map(_agent_run, [(float(e), s) for e in NOISES for s in seeds],
                             chunksize=1))
    agents = {float(e): (float(np.mean([p for ee, p, _ in runs if ee == e])),
                         float(np.mean([c for ee, _, c in runs if ee == e]))) for e in NOISES}
    (out_dir / "e22_fresh_size_raw.json").write_text(json.dumps(
        {"agents": agents, "field": field}, indent=1))
    return analyze(agents, field, _agents(), out_dir)


def analyze(agents, field, ref, out_dir: Path) -> str:
    w = np.log(N_TEST / N_FIT) / np.log(6400 / N_FIT)
    interp = {e: (1 - w) * ref[N_FIT][e][0] + w * ref[6400][e][0] for e in NOISES}
    lines = [f"Experiment 22: frozen field law (fitted at N=1600, DELTA=+{DELTA}) "
             f"at a fresh size N={N_TEST}", "",
             f"  {'noise':>6}{'phi agents':>12}{'field':>8}{'no-size':>9}{'interp.':>9}"
             f"{'chi agents':>12}{'field':>8}"]
    for e in NOISES:
        lines.append(f"  {e:6.2f}{agents[e][0]:12.3f}{field[e][0]:8.3f}{ref[N_FIT][e][0]:9.3f}"
                     f"{interp[e]:9.3f}{agents[e][1]:12.2f}{field[e][1]:8.2f}")
    err = np.array([abs(field[e][0] - agents[e][0]) for e in NOISES])
    err_base = sum(abs(ref[N_FIT][e][0] - agents[e][0]) for e in NOISES)
    err_interp = sum(abs(interp[e] - agents[e][0]) for e in NOISES)
    peak_f = float(NOISES[int(np.argmax([field[e][1] for e in NOISES]))])
    peak_a = float(NOISES[int(np.argmax([agents[e][1] for e in NOISES]))])
    f1 = int((err <= 0.05).sum()) >= 10
    f2 = abs(field[3.0][0] - agents[3.0][0]) <= 0.05
    f3 = err.sum() < err_base
    f4 = err.sum() <= 1.5 * err_interp
    f5 = abs(peak_f - peak_a) <= 0.25
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(f1)}  F1 within 0.05 at >= 10/13 [{int((err <= 0.05).sum())}/13]",
              f"  {tag(f2)}  F2 noise 3.0 within 0.05 [{field[3.0][0]:.3f} vs {agents[3.0][0]:.3f}]",
              f"  {tag(f3)}  F3 beats no size dependence [{err.sum():.3f} vs {err_base:.3f}]",
              f"  {tag(f4)}  F4 within 1.5x of log-N interpolation [{err.sum():.3f} vs "
              f"{err_interp:.3f}]",
              f"  {tag(f5)}  F5 chi peak within 0.25 [{peak_f} vs {peak_a}]"]
    _plot(agents, field, ref, interp, out_dir / "e22_fresh_size.png")
    lines.append(f"\n  figure: {out_dir / 'e22_fresh_size.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(agents, field, ref, interp, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(NOISES, [agents[e][0] for e in NOISES], "ko-", label=f"agents N={N_TEST} (new)")
    ax[0].plot(NOISES, [field[e][0] for e in NOISES], "s--", color="tab:red",
               label="frozen field law")
    ax[0].plot(NOISES, [ref[N_FIT][e][0] for e in NOISES], ":", color="tab:green",
               label="agents N=1600")
    ax[0].plot(NOISES, [ref[6400][e][0] for e in NOISES], ":", color="tab:orange",
               label="agents N=6400")
    ax[0].plot(NOISES, [interp[e] for e in NOISES], "-.", color="tab:gray",
               label="log-N interpolation")
    ax[1].plot(NOISES, [agents[e][1] for e in NOISES], "ko-", label=f"agents N={N_TEST}")
    ax[1].plot(NOISES, [field[e][1] for e in NOISES], "s--", color="tab:red",
               label="frozen field law")
    ax[0].set_title(f"Mean polarization at a fresh size N={N_TEST}")
    ax[1].set_title("Susceptibility N Var(phi)")
    for a in ax:
        a.set_xlabel("noise")
        a.grid(alpha=0.2)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
