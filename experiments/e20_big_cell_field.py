"""Experiment 20: conservative field law on big cells (~100 agents per cell).

Experiments 17-19 showed that 25-agent cells are too noisy for their averages
to obey a clean law: counting noise (sd 0.2 in density) swamps the one-step
transport signal, least squares is biased towards decay, and a hidden-state
EM fit failed validation on synthetic data. Here cells have side ELL = 5
(N0 = 100 agents on average; counting noise sd 0.1), so agents cross 0.2
cells per macro step. Lattices: 2x2 (N = 400), 4x4 (N = 1600), 8x8 (N = 6400).
The 2x2 lattice is degenerate for neighbour terms (both neighbours are the
same cell), so N = 400 is a weak test.

The law is that of experiment 18 (agents carry their momentum between cells;
local alignment, saturation, Laplacian and pressure terms; process noise),
fitted at N = 1600 only, separately for each noise level. The estimator
(ordinary least squares or instrumental variables) is chosen beforehand by
recovering a known law from synthetic data with the same counting noise
(see `validate`).

Estimator validation (synthetic 4x4 data from a known law, 4 runs x 2000
steps, counting noise sd sqrt(rho/100) persisting from step to step with
correlation 0.8, as agents cross 0.2 cells per step):
  OLS on the true hidden field recovers the law (phi 0.672 vs 0.671);
  OLS on the noisy observations over-predicts phi by ~0.07 (0.737 vs 0.671,
  0.723 vs 0.656); IV with lag 2 or 5 is erratic (phi 0.83-0.99).
OLS is therefore used. Its expected bias (~0.07) exceeds the 0.05 tolerance
of S0/T1, which may fail for that reason alone; T2-T4 test size dependence
and are less sensitive to a uniform offset.

Predictions, fixed before running (same as experiments 17-19):

  S0  At the fitted size N = 1600, mean phi within 0.05 at >= 10/13 noises.
  T1  Mean global phi within 0.05 of the agents at >= 10/13 noise levels, at
      both N = 400 and N = 6400.
  T2  At noise 3.0 the predicted order falls with size:
      phi(400) > phi(1600) > phi(6400)   (agents: 0.551 > 0.461 > 0.345).
  T3  Smaller summed |error| of mean phi than assuming no size dependence
      (using the N = 1600 agent curve), at both sizes.
  T4  Susceptibility peaks within 0.25 of the agents' peak, at both sizes.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e9_interaction_closure import BURN_IN, N_MACRO, TAU
from experiments.e11_band_closure import NOISES

ELL = 5.0
N0 = 4.0 * ELL**2
N_FIT = 1600
SEEDS = range(900, 904)


def grid_size(n: int) -> int:
    return int(round(np.sqrt(n / 4.0) / ELL))


def cell_fields(s: np.ndarray, box: float, g: int) -> np.ndarray:
    cell = np.minimum((s[:, :2] % box / box * g).astype(int), g - 1)
    idx = cell[:, 0] * g + cell[:, 1]
    rho = np.bincount(idx, minlength=g * g) / N0
    mx = np.bincount(idx, np.cos(s[:, 2]), g * g) / N0
    my = np.bincount(idx, np.sin(s[:, 2]), g * g) / N0
    return np.stack([rho, mx, my], -1).reshape(g, g, 3)


def _run(args):
    n, eta, seed = args
    from emergence_lab.systems.vicsek import Vicsek

    model = Vicsek(n_agents=n, noise=float(eta))
    g = grid_size(n)
    rng = np.random.default_rng(90_000 * n + 101 * seed + int(eta * 1000))
    s = model.initial(rng)
    for _ in range(BURN_IN):
        s = model.step(s, rng)
    out = np.empty((N_MACRO, g, g, 3), np.float32)
    for t in range(N_MACRO):
        for _ in range(TAU):
            s = model.step(s, rng)
        out[t] = cell_fields(s, model.box, g)
    return float(eta), seed, out


def simulate_fields(out_dir: Path, quick: bool = False) -> Path:
    path = out_dir / "e20_fields_1600.npz"
    if path.exists():
        return path
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    seeds = list(SEEDS)[:1] if quick else list(SEEDS)
    tasks = [(N_FIT, e, s) for e in NOISES for s in seeds]
    print(f"  simulating {len(tasks)} runs at N={N_FIT} (cells of side {ELL}) ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        sims = list(pool.map(_run, tasks, chunksize=1))
    np.savez_compressed(path, noise=np.array([s[0] for s in sims]),
                        fields=np.array([s[2] for s in sims]))
    return path


def _fit_and_predict(args):
    from experiments.e18_conservative_field import ConservativeFieldLaw

    eta, fields = args
    law = ConservativeFieldLaw(fields, iv_lag=0)
    det = law.simulate(grid_size(N_FIT), noise=False, start_aligned=True, steps=3000, burn=2999)
    out = {"noise": eta, "a": law.a.tolist(), "c": law.c, "d": law.d,
           "deterministic_phi": float(det[-1])}
    for n in (400, N_FIT, 6400):
        phi = law.simulate(grid_size(n), seed=int(eta * 100))
        out[n] = (float(phi.mean()), float(n * phi.var()))
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from experiments.e17_field_law import _agents
    from experiments.e18_conservative_field import analyze

    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(simulate_fields(out_dir, quick))
    jobs = [(float(e), d["fields"][d["noise"] == e]) for e in NOISES]
    print("  fitting and running the big-cell field law ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        preds = list(pool.map(_fit_and_predict, jobs))
    (out_dir / "e20_big_cell_field_raw.json").write_text(json.dumps(
        [{str(k): v for k, v in p.items()} for p in preds], indent=1))
    return analyze(preds, _agents(), out_dir,
                   title="Experiment 20: conservative field law on cells of side 5 "
                         "(~100 agents), fitted at N=1600",
                   name="e20_big_cell_field", free_flight=round(1.0 / ELL, 3))


if __name__ == "__main__":
    main()
