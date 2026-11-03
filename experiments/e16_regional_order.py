"""Experiment 16: a size-independent macro law from order at a fixed scale.

Experiments 14 and 15 showed that global polarization phi carries the system
size inside its law (box side ~ sqrt(N) at fixed density), and that patching
the law with a size term fails when extrapolated. Local order L, measured at a
fixed physical scale, was exactly size-independent.

So phi is replaced by an intensive variable measured at a fixed scale:

    P = regional order = sum over square regions of |sum of headings| / N,
        regions of side ~ ELL (grid g = round(box / ELL) per side), ~100 agents
        each at density 4, at every N.

Macro description z = (P, L). The macro law is fitted *without any size term*
on N = 800, 1600, 3200 and must carry over unchanged to N = 400 and 6400
(only the noise amplitude scales as sqrt(1600 / N), as measured in
experiment 14).

Predictions, fixed before running:

  R1  The size-blind (P, L) equation predicts mean P within 0.05 of the agents
      at >= 11/13 noise levels, at both N = 400 and N = 6400.
  R2  (P, L) is closed: mean autonomy of d(P) given (P, L), against the same
      20 micro features as before, is >= 0.9 over noises from 2.0 up to the
      susceptibility (N Var phi) peak, at every training size.
  R3  P is intensive in the agents themselves: at noise 3.0, mean P at N = 400
      and N = 6400 differ by < 0.05 (global phi differs by 0.22 there).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e9_interaction_closure import BLOCK, BURN_IN, N_MACRO, TAU, micro_features
from experiments.e11_band_closure import NOISES, _autonomy, _mean
from experiments.e13_macro_equation import CENTRES
from experiments.e15_size_in_law import SizedEquation

ELL = 5.0
TRAIN = {800: range(700, 704), 1600: range(710, 714), 3200: range(720, 723)}
TEST = {400: range(730, 734), 6400: range(740, 743)}


def regional_order(pos: np.ndarray, theta: np.ndarray, box: float) -> float:
    g = max(1, int(round(box / ELL)))
    cell = np.minimum((pos / box * g).astype(int), g - 1)
    idx = cell[:, 0] * g + cell[:, 1]
    c = np.bincount(idx, np.cos(theta), g * g)
    s = np.bincount(idx, np.sin(theta), g * g)
    return float(np.hypot(c, s).sum() / len(theta))


def _run(args):
    n, eta, seed = args
    from emergence_lab.systems.vicsek import Vicsek

    model = Vicsek(n_agents=n, noise=float(eta))
    rng = np.random.default_rng(50_000 * n + 101 * seed + int(eta * 1000))
    s = model.initial(rng)
    for _ in range(BURN_IN):
        s = model.step(s, rng)
    phi, reg, feats = np.empty(N_MACRO), np.empty(N_MACRO), []
    for t in range(N_MACRO):
        for _ in range(TAU):
            s = model.step(s, rng)
        phi[t] = np.hypot(np.cos(s[:, 2]).mean(), np.sin(s[:, 2]).mean())
        reg[t] = regional_order(s[:, :2] % model.box, s[:, 2], model.box)
        feats.append(micro_features(s, model.box))
    return n, float(eta), seed, phi, reg, np.array(feats)


def _simulate_all(out_dir: Path, quick: bool) -> dict:
    path = out_dir / "e16_trajectories.npz"
    if not path.exists():
        sizes = {**TRAIN, **TEST}
        tasks = [(n, e, s) for n in sorted(sizes, reverse=True) for e in NOISES
                 for s in (list(sizes[n])[:1] if quick else sizes[n])]
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = "1"
        print(f"  simulating {len(tasks)} runs ...", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
            sims = list(pool.map(_run, tasks, chunksize=1))
        np.savez_compressed(path, n=np.array([s[0] for s in sims]),
                            noise=np.array([s[1] for s in sims]),
                            seed=np.array([s[2] for s in sims]),
                            phi=np.array([s[3] for s in sims]),
                            reg=np.array([s[4] for s in sims]),
                            feats=np.array([s[5] for s in sims]))
    return dict(np.load(path))


def _closure(args):
    reg, local, feats = args
    return _autonomy(np.column_stack([reg, local]), np.diff(reg), feats)


def analyze(d: dict, out_dir: Path) -> str:
    local = d["feats"][:, :, :8] @ CENTRES
    sizes = sorted(set(d["n"].tolist()))

    def mean_at(key_arr, n, e):
        return float(key_arr[(d["n"] == n) & (d["noise"] == e)].mean())

    runs = [(int(n), float(e), np.column_stack([r, l]))
            for n, e, r, l in zip(d["n"], d["noise"], d["reg"], local) if int(n) in TRAIN]
    eq = SizedEquation(runs, size_term=False, cap_order=False)

    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        auton = list(pool.map(_closure, zip(d["reg"], local, d["feats"]), chunksize=4))
    auton = np.array(auton)

    lines = [f"Experiment 16: regional order (scale {ELL}) + local order, one law for all sizes",
             "", "  Agents: mean global phi and mean regional order P",
             "  " + f"{'noise':>6}" + "".join(f"{f'phi {n}':>10}" for n in sizes)
             + "".join(f"{f'P {n}':>9}" for n in sizes)]
    for e in NOISES:
        lines.append("  " + f"{e:6.2f}" + "".join(f"{mean_at(d['phi'], n, e):10.3f}" for n in sizes)
                     + "".join(f"{mean_at(d['reg'], n, e):9.3f}" for n in sizes))

    pred, hits, chi_peak, closure = {}, {}, {}, {}
    for n in TEST:
        pred[n] = {e: float(eq.simulate(n, e).mean()) for e in NOISES}
        hits[n] = int(sum(abs(pred[n][e] - mean_at(d["reg"], n, e)) <= 0.05 for e in NOISES))
    lines += ["", "  Size-blind (P, L) equation, fitted on N = 800, 1600, 3200:",
              "  " + f"{'noise':>6}" + "".join(f"{f'P {n} agents':>15}{f'P {n} eq.':>11}"
                                              for n in TEST)]
    for e in NOISES:
        lines.append("  " + f"{e:6.2f}" + "".join(
            f"{mean_at(d['reg'], n, e):15.3f}{pred[n][e]:11.3f}" for n in TEST))

    lines += ["", "  Closure: autonomy of d(P) given (P, L), noise 2.0 .. chi peak"]
    for n in sizes:
        chi = [float(np.mean([n * p.var() for p in d["phi"][(d["n"] == n) & (d["noise"] == e)]]))
               for e in NOISES]
        ic = int(np.argmax(chi))
        chi_peak[n] = float(NOISES[ic])
        per_noise = [_mean(auton[(d["n"] == n) & (d["noise"] == e)]) for e in NOISES]
        closure[n] = _mean(per_noise[: ic + 1])
        lines.append(f"    N={n:5d}: chi peak {chi_peak[n]}, mean autonomy {closure[n]:.3f}, "
                     f"at peak {per_noise[ic]:.3f}")

    r1 = all(h >= 11 for h in hits.values())
    r2 = all(np.isfinite(closure[n]) and closure[n] >= 0.9 for n in TRAIN)
    gap_p = abs(mean_at(d["reg"], 400, 3.0) - mean_at(d["reg"], 6400, 3.0))
    gap_phi = abs(mean_at(d["phi"], 400, 3.0) - mean_at(d["phi"], 6400, 3.0))
    r3 = gap_p < 0.05
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(r1)}  R1 size-blind equation: mean P within 0.05 at >= 11/13 "
              + ", ".join(f"[N={n}: {h}/13]" for n, h in hits.items()),
              f"  {tag(r2)}  R2 (P, L) closed at every training size, mean autonomy >= 0.9 "
              + ", ".join(f"[N={n}: {closure[n]:.3f}]" for n in TRAIN),
              f"  {tag(r3)}  R3 P intensive at noise 3.0: |P(400) - P(6400)| < 0.05 "
              f"[{gap_p:.3f}; global phi differs by {gap_phi:.3f}]"]
    _plot(d, pred, out_dir / "e16_regional_order.png")
    lines.append(f"\n  figure: {out_dir / 'e16_regional_order.png'}")
    (out_dir / "e16_regional_order_raw.json").write_text(json.dumps(
        {"pred": {str(n): {str(e): v for e, v in p.items()} for n, p in pred.items()},
         "closure": {str(n): v for n, v in closure.items()}, "chi_peak": chi_peak}, indent=1))
    text = "\n".join(lines)
    print(text)
    return text


def _plot(d, pred, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(set(d["n"].tolist()))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, n in enumerate(sizes):
        c = cmap(k / max(1, len(sizes) - 1))
        for i, key in enumerate(("phi", "reg")):
            m = [d[key][(d["n"] == n) & (d["noise"] == e)].mean() for e in NOISES]
            ax[i].plot(NOISES, m, "o-", color=c, ms=4, label=f"agents N={n}")
        if n in pred:
            ax[1].plot(NOISES, [pred[n][e] for e in NOISES], "s--", color=c, alpha=0.7,
                       label=f"equation N={n} (unseen size)")
    ax[0].set_title("Global polarization: depends on size")
    ax[1].set_title(f"Regional order P (scale {ELL}): one law for all sizes?")
    for a in ax:
        a.set_xlabel("noise")
        a.grid(alpha=0.2)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    return analyze(_simulate_all(out_dir, quick), out_dir)


if __name__ == "__main__":
    main()
