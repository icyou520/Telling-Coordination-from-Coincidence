"""Experiment 21: one-parameter calibration of the big-cell field law.

Experiment 20's field law (cells of ~100 agents, fitted at N = 1600) ordered,
had a transition, and produced size dependence in the right direction, but
under-ordered at every size by a similar amount, as if the agents' curves were
shifted to lower noise. Hypothesis: the law's structure, including how size
enters, is right; one global strength is mis-estimated (counting-noise bias).

One parameter, the same at every noise level: an offset DELTA added to the
fitted momentum-decay coefficient a1 (positive = less decay relative to
alignment). DELTA is chosen on a grid to minimise the summed |error| of mean
phi at the fitted size N = 1600 only (agents of experiment 16), then frozen.
Everything else is experiment 20's law unchanged.

Predictions, fixed before running:

  T1  Mean global phi within 0.05 of the agents at >= 10/13 noise levels, at
      both N = 400 and N = 6400.
  T2  At noise 3.0 the predicted order falls with size:
      phi(400) > phi(1600) > phi(6400).
  T3  Smaller summed |error| of mean phi than assuming no size dependence
      (using the N = 1600 agent curve), at both sizes.
  T4  Susceptibility peaks within 0.25 of the agents' peak, at both sizes.
  T5  The calibrated law has a smaller summed |error| than experiment 20's
      uncalibrated law, at both N = 400 and N = 6400.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e11_band_closure import NOISES
from experiments.e20_big_cell_field import ELL, N_FIT, grid_size, simulate_fields

DELTAS = np.round(np.arange(-0.02, 0.0801, 0.01), 3)
CAL_STEPS, CAL_BURN = 10_000, 1_000


def _law(eta, fields, delta):
    from experiments.e18_conservative_field import ConservativeFieldLaw

    law = ConservativeFieldLaw(fields, iv_lag=0)
    law.a = law.a.copy()
    law.a[0] += delta
    return law


def _calibration_job(args):
    eta, fields, delta = args
    phi = _law(eta, fields, delta).simulate(grid_size(N_FIT), seed=int(eta * 100),
                                            steps=CAL_STEPS, burn=CAL_BURN)
    return eta, delta, float(phi.mean())


def _predict_job(args):
    eta, fields, delta = args
    law = _law(eta, fields, delta)
    out = {"noise": eta, "delta": delta, "c": law.c, "d": law.d, "a": law.a.tolist()}
    for n in (400, N_FIT, 6400):
        phi = law.simulate(grid_size(n), seed=int(eta * 100) + 1)
        out[n] = (float(phi.mean()), float(n * phi.var()))
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from experiments.e17_field_law import _agents

    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(simulate_fields(out_dir, quick))
    fields = {float(e): d["fields"][d["noise"] == e] for e in NOISES}
    agents = _agents()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"  calibrating DELTA over {len(DELTAS)} values at N={N_FIT} ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cal = list(pool.map(_calibration_job, [(e, fields[e], float(dl)) for dl in DELTAS
                                               for e in fields], chunksize=2))
    cost = {float(dl): sum(abs(p - agents[N_FIT][e][0]) for e, dd, p in cal if dd == dl)
            for dl in DELTAS}
    delta = min(cost, key=cost.get)
    print(f"  chosen DELTA = {delta} (summed |error| at N={N_FIT}: {cost[delta]:.3f})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        preds = list(pool.map(_predict_job, [(e, fields[e], delta) for e in fields]))
    (out_dir / "e21_calibrated_field_raw.json").write_text(json.dumps(
        {"cost": cost, "delta": delta,
         "preds": [{str(k): v for k, v in p.items()} for p in preds]}, indent=1))
    return analyze(preds, agents, cost, delta, out_dir)


def analyze(preds, agents, cost, delta, out_dir: Path) -> str:
    preds = {p["noise"]: p for p in preds}
    raw20 = json.loads((out_dir / "e20_big_cell_field_raw.json").read_text())
    e20 = {p["noise"]: {int(k): v for k, v in p.items() if k.isdigit()} for p in raw20}
    sizes = (400, N_FIT, 6400)
    lines = [f"Experiment 21: big-cell field law with one calibrated offset (cells of side {ELL})",
             "", "  calibration at N=1600 (summed |error| of mean phi by DELTA): "
             + "  ".join(f"{k:+.2f}:{v:.2f}" for k, v in cost.items()),
             f"  chosen DELTA = {delta:+.2f}", "",
             f"  {'noise':>6}" + "".join(f"{f'phi {n} ag.':>13}{'field':>8}" for n in sizes)
             + "".join(f"{f'chi {n} ag.':>13}{'field':>8}" for n in (400, 6400))]
    for e in NOISES:
        lines.append(f"  {e:6.2f}" + "".join(f"{agents[n][e][0]:13.3f}{preds[e][n][0]:8.3f}"
                                             for n in sizes)
                     + "".join(f"{agents[n][e][1]:13.2f}{preds[e][n][1]:8.2f}"
                               for n in (400, 6400)))
    hits, err, base, peak, err20 = {}, {}, {}, {}, {}
    for n in sizes:
        e_field = np.array([abs(preds[e][n][0] - agents[n][e][0]) for e in NOISES])
        hits[n], err[n] = int((e_field <= 0.05).sum()), e_field.sum()
        base[n] = sum(abs(agents[N_FIT][e][0] - agents[n][e][0]) for e in NOISES)
        err20[n] = sum(abs(e20[e][n][0] - agents[n][e][0]) for e in NOISES)
        peak[n] = (float(NOISES[int(np.argmax([preds[e][n][1] for e in NOISES]))]),
                   float(NOISES[int(np.argmax([agents[n][e][1] for e in NOISES]))]))
    p3 = {n: preds[3.0][n][0] for n in sizes}
    new = (400, 6400)
    t1 = all(hits[n] >= 10 for n in new)
    t2 = p3[400] > p3[N_FIT] > p3[6400]
    t3 = all(err[n] < base[n] for n in new)
    t4 = all(abs(peak[n][0] - peak[n][1]) <= 0.25 for n in new)
    t5 = all(err[n] < err20[n] for n in new)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", f"  calibrated size N=1600: within 0.05 at {hits[N_FIT]}/13, "
                  f"chi peak {peak[N_FIT][0]} vs agents {peak[N_FIT][1]}",
              "", "  Pre-registered predictions (unseen sizes):",
              f"  {tag(t1)}  T1 within 0.05 at >= 10/13 "
              + ", ".join(f"[N={n}: {hits[n]}/13]" for n in new),
              f"  {tag(t2)}  T2 order falls with size at noise 3.0 "
              f"[{p3[400]:.3f} > {p3[N_FIT]:.3f} > {p3[6400]:.3f}]",
              f"  {tag(t3)}  T3 beats 'no size dependence' "
              + ", ".join(f"[N={n}: {err[n]:.3f} vs {base[n]:.3f}]" for n in new),
              f"  {tag(t4)}  T4 chi peak within 0.25 "
              + ", ".join(f"[N={n}: {peak[n][0]} vs {peak[n][1]}]" for n in new),
              f"  {tag(t5)}  T5 beats the uncalibrated law "
              + ", ".join(f"[N={n}: {err[n]:.3f} vs {err20[n]:.3f}]" for n in new)]
    _plot(preds, agents, out_dir / "e21_calibrated_field.png")
    lines.append(f"\n  figure: {out_dir / 'e21_calibrated_field.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(preds, agents, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {400: "tab:purple", N_FIT: "tab:green", 6400: "tab:orange"}
    for n, c in colors.items():
        for i in range(2):
            ax[i].plot(NOISES, [agents[n][e][i] for e in NOISES], "o-", color=c,
                       label=f"agents N={n}")
            ax[i].plot(NOISES, [preds[e][n][i] for e in NOISES], "s--", color=c, alpha=0.7,
                       label=f"field law N={n}" + (" (calibrated)" if n == N_FIT else ""))
    ax[0].set_title("Global polarization: field law, one offset calibrated at N=1600")
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
