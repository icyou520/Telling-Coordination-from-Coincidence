"""Experiment 14: does the flock's macro equation predict flocks of other sizes?

Experiment 13 fitted z_{t+1} = z_t + f(z_t, eta) + sigma(z_t, eta) xi for
z = (phi, L) at N = 1600; with the noise amplitude calibrated (a post-hoc fix)
it reproduced the order curve and the transition on held-out noise levels.

Here that equation is tested prospectively on sizes it has never seen. It is
fitted once on all N = 1600 trajectories (experiments 11 and 12) and carried
to size N with a single theory-given change:

    drift f unchanged (alignment is a local, intensive effect),
    noise sigma scaled by sqrt(1600 / N) (fluctuations of means of N agents).

The agents' transition moves with N (experiment 10: noise 3.75 at N = 400,
3.25 at N = 1600) and its susceptibility peak grows with N. Neither is put in
by hand.

A readable version keeps 6 terms per variable in the drift (greedy forward
selection on the training data); its noise model is the same.

Predictions, fixed before running (N = 800 and N = 3200, fresh agents):

  P1  Full equation: mean phi within 0.05 of the agents at >= 11 of 13 noise
      levels, at both sizes.
  P2  Full equation: susceptibility peak within 0.25 of the agents' peak, at
      both sizes.
  P3  Full equation: peak susceptibility within a factor 2, at both sizes.
  P4  6-term equation: P1 and P2 hold for it as well.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e11_band_closure import NOISES, _simulate
from experiments.e13_macro_equation import CENTRES, MacroEquation, feature_names

N_FIT = 1600
SIZES = {800: range(200, 206), 3200: range(300, 304)}
N_TERMS = 6


def _load(paths):
    noise, phi, local = [], [], []
    for p in paths:
        d = np.load(p)
        noise.append(d["noise"])
        phi.append(d["phi"])
        local.append(d["feats"][:, :, :8] @ CENTRES)
    return np.concatenate(noise), np.concatenate(phi), np.concatenate(local)


def _agents(out_dir: Path, quick: bool):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    result = {}
    for n, seeds in SIZES.items():
        path = out_dir / f"e14_trajectories_{n}.npz"
        if not path.exists():
            seeds = list(seeds)[:1] if quick else list(seeds)
            tasks = [(n, e, s, "flock") for e in NOISES for s in seeds]
            print(f"  simulating N={n}: {len(tasks)} runs ...", flush=True)
            with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
                sims = list(pool.map(_simulate, tasks, chunksize=1))
            np.savez_compressed(path, noise=np.array([s[0] for s in sims]),
                                seed=np.array([s[1] for s in sims]),
                                phi=np.array([s[2] for s in sims]),
                                feats=np.array([s[3] for s in sims]))
        d = np.load(path)
        result[n] = {e: (float(np.mean([p.mean() for p in d["phi"][d["noise"] == e]])),
                         float(np.mean([n * p.var() for p in d["phi"][d["noise"] == e]])))
                     for e in NOISES}
    return result


def _predict(eq: MacroEquation, n: int):
    out = {}
    for e in NOISES:
        phi = eq.simulate(np.array([0.5, 0.65]), e, noise_scale=np.sqrt(N_FIT / n))[:, 0]
        out[e] = (float(phi.mean()), float(n * phi.var()))
    return out


def _format_drift(eq: MacroEquation) -> list[str]:
    names = feature_names(["phi", "L"])
    lines = []
    for c, var in enumerate(("phi", "L")):
        terms = eq.raw_drift_terms(names)[c]
        body = " ".join(f"{'+' if coef >= 0 else '-'} {abs(coef):.4g}"
                        + ("" if name == "1" else f" {name}") for coef, name in terms)
        lines.append(f"    d{var} = {body.lstrip('+ ')}")
    return lines


def analyze(agents, models, out_dir: Path) -> str:
    lines = ["Experiment 14: macro equation fitted at N=1600, predicting other sizes", ""]
    lines += ["  6-term drift per macro step (10 agent steps), fitted at N=1600:"]
    lines += _format_drift(models["6-term"])
    verdict = {}
    preds = {}
    for n in SIZES:
        for name, eq in models.items():
            preds[(n, name)] = _predict(eq, n)
        a = agents[n]
        peak_a = float(NOISES[int(np.argmax([a[e][1] for e in NOISES]))])
        lines += ["", f"  N = {n}",
                  f"  {'noise':>6}{'phi agents':>12}{'phi full':>10}{'phi 6-term':>12}"
                  f"{'chi agents':>12}{'chi full':>10}{'chi 6-term':>12}"]
        pf, ps = preds[(n, "full")], preds[(n, "6-term")]
        for e in NOISES:
            lines.append(f"  {e:6.2f}{a[e][0]:12.3f}{pf[e][0]:10.3f}{ps[e][0]:12.3f}"
                         f"{a[e][1]:12.2f}{pf[e][1]:10.2f}{ps[e][1]:12.2f}")
        for name in models:
            p = preds[(n, name)]
            hits = int(sum(abs(p[e][0] - a[e][0]) <= 0.05 for e in NOISES))
            peak = float(NOISES[int(np.argmax([p[e][1] for e in NOISES]))])
            ratio = max(p[e][1] for e in NOISES) / max(a[e][1] for e in NOISES)
            verdict[(n, name)] = (hits, peak, peak_a, ratio)
            lines.append(f"  {name:>7}: mean phi within 0.05 at {hits}/13, chi peak {peak} "
                         f"(agents {peak_a}), peak chi ratio {ratio:.2f}")

    def ok_p1(name):
        return all(verdict[(n, name)][0] >= 11 for n in SIZES)

    def ok_p2(name):
        return all(abs(verdict[(n, name)][1] - verdict[(n, name)][2]) <= 0.25 for n in SIZES)

    p3 = all(0.5 <= verdict[(n, "full")][3] <= 2.0 for n in SIZES)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(ok_p1('full'))}  P1 full: mean phi within 0.05 at >= 11/13, both sizes",
              f"  {tag(ok_p2('full'))}  P2 full: chi peak within 0.25, both sizes",
              f"  {tag(p3)}  P3 full: peak chi within a factor 2, both sizes",
              f"  {tag(ok_p1('6-term') and ok_p2('6-term'))}  P4 6-term: P1 and P2 hold"]
    _plot(agents, preds, out_dir / "e14_equation_new_sizes.png")
    lines.append(f"\n  figure: {out_dir / 'e14_equation_new_sizes.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(agents, preds, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {800: "tab:blue", 3200: "tab:orange"}
    for n in SIZES:
        for i in range(2):
            ax[i].plot(NOISES, [agents[n][e][i] for e in NOISES], "o-", color=colors[n],
                       label=f"agents N={n}")
            ax[i].plot(NOISES, [preds[(n, "full")][e][i] for e in NOISES], "s--",
                       color=colors[n], alpha=0.7, label=f"equation N={n}")
            ax[i].plot(NOISES, [preds[(n, "6-term")][e][i] for e in NOISES], ":",
                       color=colors[n], alpha=0.7, label=f"6-term N={n}")
    ax[0].set_title("Mean polarization (equation fitted at N=1600 only)")
    ax[1].set_title("Susceptibility N Var(phi)")
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
    train = [out_dir / "e11_trajectories.npz", out_dir / "e12_trajectories.npz"]
    if not all(p.exists() for p in train):
        raise SystemExit("needs e11/e12 trajectories: run `run.py bands local` first")
    noise, phi, local = _load(train)
    z = [np.column_stack([p, l]) for p, l in zip(phi, local)]
    models = {"full": MacroEquation(noise, z, calibrate_noise=True),
              "6-term": MacroEquation(noise, z, calibrate_noise=True, n_terms=N_TERMS)}
    agents = _agents(out_dir, quick)
    return analyze(agents, models, out_dir)


if __name__ == "__main__":
    main()
