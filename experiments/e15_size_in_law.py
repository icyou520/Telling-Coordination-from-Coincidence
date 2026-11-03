"""Experiment 15: one macro equation for every flock size.

Experiment 14 found that, for z = (phi, L), the macro law G has the same form
at N = 800, 1600 and 3200, the noise scales exactly as 1/sqrt(N), local order
L is size-independent, and only the drift of phi depends on N. Pure 1/N
scaling of that drift cannot be right (with drift and noise variance both
~1/N, the stationary distribution of phi would not move, but the agents' does),
and the noise-induced drift of a vector modulus is ~8x too small.

Hypothesis tested: size enters the macro law in exactly one place, the drift
of phi, linearly in x = log2(N / 1600):

    d(phi) = f0(phi, L, eta) + x f1(phi, L, eta)
    d(L)   = g(phi, L, eta)
    noise  = sigma(phi, L, eta) sqrt(1600 / N)

with f0, f1, g, log sigma^2 in the 30-term basis of experiment 13. Fitted on
N = 800, 1600, 3200 (x = -1, 0, 1); tested on new sizes N = 400 (x = -2) and
N = 6400 (x = +2), i.e. one doubling beyond the fitted range on each side.

Control: the same equation without the x term, fitted on the same data.

Predictions, fixed before running:

  Q1  Mean phi within 0.05 of the agents at >= 11/13 noise levels, at both sizes.
  Q2  Susceptibility peak within 0.25 of the agents' peak, at both sizes.
  Q3  Peak susceptibility within a factor 2, at both sizes.
  Q4  The size-blind control has a larger summed |error| of mean phi at both sizes.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e11_band_closure import NOISES, _simulate
from experiments.e13_macro_equation import BURN, CENTRES, STEPS, features

N_REF = 1600
TEST = {400: range(400, 406), 6400: range(600, 604)}


def _load(path: Path, n: int):
    d = np.load(path)
    return [(n, float(e), np.column_stack([p, f[:, :8] @ CENTRES]))
            for e, p, f in zip(d["noise"], d["phi"], d["feats"])]


def _ridge(x, y, ridge=1e-8):
    return np.linalg.solve(x.T @ x + ridge * len(x) * np.eye(x.shape[1]), x.T @ y)


class SizedEquation:
    def __init__(self, runs, size_term: bool = True, cap_order: bool = True):
        self.size_term, self.cap_order = size_term, cap_order
        z0, dz, eta, xs = [], [], [], []
        for n, e, z in runs:
            z0.append(z[:-1])
            dz.append(np.diff(z, axis=0))
            eta.append(np.full(len(z) - 1, e))
            xs.append(np.full(len(z) - 1, np.log2(n / N_REF)))
        z0, dz, eta, xs = np.vstack(z0), np.vstack(dz), np.concatenate(eta), np.concatenate(xs)
        f = features(z0, eta)
        self.mu, self.sd = f.mean(0), f.std(0)
        self.mu[0], self.sd[0] = 0.0, 1.0
        self.sd[self.sd == 0] = 1.0
        fs = (f - self.mu) / self.sd
        self.w_phi = _ridge(self._phi_design(fs, xs), dz[:, 0])
        self.w_l = _ridge(fs, dz[:, 1])
        res = dz - np.column_stack([self._phi_design(fs, xs) @ self.w_phi, fs @ self.w_l])
        scale = np.sqrt(np.exp2(xs))[:, None]
        res_ref = res * scale
        self.w_log = _ridge(fs, np.log(res_ref**2 + 1e-12))
        self.w_log[0] += np.log(np.mean(res_ref**2 / np.exp(fs @ self.w_log), axis=0))
        sig = np.exp(0.5 * (fs @ self.w_log))
        corr = np.corrcoef((res_ref / sig).T)
        self.chol = np.linalg.cholesky(corr + 1e-9 * np.eye(2))

    def _phi_design(self, fs, xs):
        return np.hstack([fs, fs * xs[:, None]]) if self.size_term else fs

    def simulate(self, n: int, eta: float, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.log2(n / N_REF)
        z = np.array([0.5, 0.65])
        out = np.empty(STEPS)
        for t in range(STEPS):
            fs = (features(z[None], eta) - self.mu) / self.sd
            drift = np.array([(self._phi_design(fs, np.array([x])) @ self.w_phi)[0],
                              (fs @ self.w_l)[0]])
            sig = np.exp(0.5 * (fs @ self.w_log))[0] * np.sqrt(N_REF / n)
            z = np.clip(z + drift + sig * (self.chol @ rng.standard_normal(2)), 0.0, 1.0)
            if self.cap_order:
                z[0] = min(z[0], z[1])
            out[t] = z[0]
        return out[BURN:]


def _agents(out_dir: Path, quick: bool):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    result = {}
    for n, seeds in TEST.items():
        path = out_dir / f"e15_trajectories_{n}.npz"
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


def analyze(agents, models, out_dir: Path) -> str:
    lines = ["Experiment 15: one macro equation for all flock sizes "
             "(fitted on N = 800, 1600, 3200)"]
    preds, score = {}, {}
    for n in TEST:
        a = agents[n]
        for name, eq in models.items():
            p = {}
            for e in NOISES:
                phi = eq.simulate(n, e)
                p[e] = (float(phi.mean()), float(n * phi.var()))
            preds[(n, name)] = p
        peak_a = float(NOISES[int(np.argmax([a[e][1] for e in NOISES]))])
        lines += ["", f"  N = {n}",
                  f"  {'noise':>6}{'phi agents':>12}{'phi eq.':>9}{'phi blind':>11}"
                  f"{'chi agents':>12}{'chi eq.':>9}{'chi blind':>11}"]
        ps, pb = preds[(n, "sized")], preds[(n, "blind")]
        for e in NOISES:
            lines.append(f"  {e:6.2f}{a[e][0]:12.3f}{ps[e][0]:9.3f}{pb[e][0]:11.3f}"
                         f"{a[e][1]:12.2f}{ps[e][1]:9.2f}{pb[e][1]:11.2f}")
        for name in models:
            p = preds[(n, name)]
            err = np.array([abs(p[e][0] - a[e][0]) for e in NOISES])
            peak = float(NOISES[int(np.argmax([p[e][1] for e in NOISES]))])
            ratio = max(p[e][1] for e in NOISES) / max(a[e][1] for e in NOISES)
            score[(n, name)] = (int((err <= 0.05).sum()), peak, peak_a, ratio, float(err.sum()))
            lines.append(f"  {name:>6}: within 0.05 at {score[(n, name)][0]}/13, chi peak {peak} "
                         f"(agents {peak_a}), peak chi ratio {ratio:.2f}, "
                         f"summed |error| {err.sum():.3f}")
    s = {n: score[(n, "sized")] for n in TEST}
    q1 = all(v[0] >= 11 for v in s.values())
    q2 = all(abs(v[1] - v[2]) <= 0.25 for v in s.values())
    q3 = all(0.5 <= v[3] <= 2.0 for v in s.values())
    q4 = all(score[(n, "blind")][4] > score[(n, "sized")][4] for n in TEST)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(q1)}  Q1 mean phi within 0.05 at >= 11/13, both sizes "
              + ", ".join(f"[N={n}: {v[0]}/13]" for n, v in s.items()),
              f"  {tag(q2)}  Q2 chi peak within 0.25, both sizes "
              + ", ".join(f"[N={n}: {v[1]} vs {v[2]}]" for n, v in s.items()),
              f"  {tag(q3)}  Q3 peak chi within a factor 2, both sizes "
              + ", ".join(f"[N={n}: {v[3]:.2f}]" for n, v in s.items()),
              f"  {tag(q4)}  Q4 size-blind control is worse, both sizes "
              + ", ".join(f"[N={n}: {score[(n, 'blind')][4]:.3f} vs {score[(n, 'sized')][4]:.3f}]"
                          for n in TEST)]
    _plot(agents, preds, out_dir / "e15_size_in_law.png")
    lines.append(f"\n  figure: {out_dir / 'e15_size_in_law.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(agents, preds, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {400: "tab:blue", 6400: "tab:orange"}
    for n in TEST:
        for i in range(2):
            ax[i].plot(NOISES, [agents[n][e][i] for e in NOISES], "o-", color=colors[n],
                       label=f"agents N={n}")
            ax[i].plot(NOISES, [preds[(n, "sized")][e][i] for e in NOISES], "s--",
                       color=colors[n], alpha=0.75, label=f"equation N={n}")
            ax[i].plot(NOISES, [preds[(n, "blind")][e][i] for e in NOISES], ":",
                       color=colors[n], alpha=0.6, label=f"size-blind N={n}")
    ax[0].set_title("Mean polarization at unseen sizes")
    ax[1].set_title("Susceptibility N Var(phi) at unseen sizes")
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
    needed = {800: out_dir / "e14_trajectories_800.npz",
              1600: out_dir / "e12_trajectories.npz",
              3200: out_dir / "e14_trajectories_3200.npz"}
    if not all(p.exists() for p in needed.values()):
        raise SystemExit("needs trajectories from experiments 12 and 14 (run.py local sizes)")
    runs = [r for n, p in needed.items() for r in _load(p, n)]
    runs += _load(out_dir / "e11_trajectories.npz", 1600)
    models = {"sized": SizedEquation(runs, size_term=True),
              "blind": SizedEquation(runs, size_term=False)}
    agents = _agents(out_dir, quick)
    return analyze(agents, models, out_dir)


if __name__ == "__main__":
    main()
