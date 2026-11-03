"""Experiment 18: field law with momentum carried by the agents.

Experiment 17's field law decayed to disorder even without noise: agents move
0.4 cells per macro step, so cell momentum looks like it decays, but it is
really carried into neighbouring cells, which the fitted law did not do.

Here every extensive cell quantity q (density rho and momentum m) moves with
the agents. Each macro step, cell i sends to each neighbour the fraction

    f_dir = c max(+-u_k, 0) + D,      u = m / rho  (mean heading in the cell)

of its agents and of their momentum (positivity-preserving, exactly
conservative). c and D >= 0 are fitted on the density data only. The
remaining, local part of the momentum update is fitted on what transport
does not explain:

    m_next - T(m) = (a1 + a2 rho + a3 |m|^2 + a4 rho^2) m
                    + a5 Lap m + a6 grad rho + noise

Cell momentum is measured from ~25 agents whose membership changes every
step, so ordinary least squares is biased towards decay (errors in the
regressors; with it, an aligned flock decays to disorder without noise).
The local coefficients are therefore estimated by instrumental variables,
using the same features IV_LAG = 2 macro steps earlier as instruments
(chosen on training data only: lag 2 keeps an aligned noise-free flock
ordered with stable coefficients; lag 4 gives erratic coefficients).

Training data, lattices and noise model are those of experiment 17
(N = 1600, 8 x 8 cells of side 2.5), and the guards are unchanged.

Predictions, fixed before running (identical to experiment 17, plus a check
at the fitted size):

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

from experiments.e11_band_closure import NOISES
from experiments.e17_field_law import BURN, N_FIT, STEPS, _agents, _dx, _lap_axes, grid_size

IV_LAG = 2


def _velocity(rho, m):
    return m / np.maximum(rho, 1e-9)[..., None]


def _fractions(rho, m, c, d):
    u = _velocity(rho, m)
    return [c * np.maximum(s * u[..., k], 0) + d for k in (0, 1) for s in (1, -1)]


def transport(q, fracs, ax0, ax1):
    """Move a share fracs[dir] of q to each neighbour: (+ax0, -ax0, +ax1, -ax1)."""
    expand = (lambda f: f[..., None]) if q.ndim > fracs[0].ndim else (lambda f: f)
    out = q * expand(sum(fracs))
    inflow = (np.roll(q * expand(fracs[0]), 1, ax0) + np.roll(q * expand(fracs[1]), -1, ax0)
              + np.roll(q * expand(fracs[2]), 1, ax1) + np.roll(q * expand(fracs[3]), -1, ax1))
    return q - out + inflow


def _transport_parts(q, rho, m, ax0, ax1):
    """T(q) - q = c A + D B; returns (A, B)."""
    a = transport(q, _fractions(rho, m, 1.0, 0.0), ax0, ax1) - q
    b = transport(q, _fractions(rho, m, 0.0, 1.0), ax0, ax1) - q
    return a, b


def local_features(rho, m, ax0, ax1):
    m2 = (m**2).sum(-1)
    grad = np.stack([_dx(rho, ax0), _dx(rho, ax1)], -1)
    lap = np.stack([_lap_axes(m[..., k], ax0, ax1) for k in (0, 1)], -1)
    return [m, rho[..., None] * m, m2[..., None] * m, rho[..., None] ** 2 * m, lap, grad]


class ConservativeFieldLaw:
    def __init__(self, fields: np.ndarray, iv_lag: int | None = None):
        lag = IV_LAG if iv_lag is None else iv_lag
        f = fields.astype(float)
        rho, m = f[:, :-1, ..., 0], f[:, :-1, ..., 1:]
        rho1, m1 = f[:, 1:, ..., 0], f[:, 1:, ..., 1:]
        ax0, ax1 = rho.ndim - 2, rho.ndim - 1
        ar, br = _transport_parts(rho, rho, m, ax0, ax1)
        y = (rho1 - rho).reshape(-1)
        c, d = np.linalg.lstsq(np.stack([ar.reshape(-1), br.reshape(-1)], 1), y, rcond=None)[0]
        if d < 0:
            d = 0.0
            c = float((ar.reshape(-1) @ y) / (ar.reshape(-1) @ ar.reshape(-1)))
        self.c, self.d = float(c), float(d)
        resid_rho = y - (self.c * ar + self.d * br).reshape(-1)
        self.flux_sd = float(np.sqrt(resid_rho.var() / 4))

        moved = transport(m, _fractions(rho, m, self.c, self.d), ax0, ax1)
        target_all = m1 - moved
        x_all = np.stack(local_features(rho, m, ax0, ax1), -1)
        x = x_all[:, lag:].reshape(-1, x_all.shape[-1])
        z = x_all[:, : x_all.shape[1] - lag].reshape(-1, x_all.shape[-1])
        target = target_all[:, lag:].reshape(-1)
        self.a = np.linalg.solve(z.T @ x, z.T @ target)
        res = (target - x @ self.a).reshape(target_all[:, lag:].shape)
        rho, m = rho[:, lag:], m[:, lag:]
        m2 = (m**2).sum(-1)
        xs = np.stack([np.ones(rho.size), rho.reshape(-1), m2.reshape(-1)], 1)
        r2 = (res**2).mean(-1).reshape(-1)
        self.b = np.linalg.lstsq(xs, np.log(r2 + 1e-12), rcond=None)[0]
        self.b[0] += np.log(np.mean(r2 / np.exp(xs @ self.b)))
        self.rho_range = (float(rho.min()), float(rho.max()))
        self.m2_max = float(m2.max())

    def simulate(self, g: int, seed: int = 0, noise: bool = True, start_aligned: bool = False,
                 steps: int = STEPS, burn: int = BURN) -> np.ndarray:
        rng = np.random.default_rng(seed)
        rho = np.ones((g, g))
        ang = np.zeros((g, g)) if start_aligned else rng.uniform(-np.pi, np.pi, (g, g))
        m = (0.9 if start_aligned else 0.3) * np.stack([np.cos(ang), np.sin(ang)], -1)
        mass = rho.sum()
        a1, a2, a3, a4, a5, a6 = self.a
        phi = np.empty(steps)
        for t in range(steps):
            fr = _fractions(rho, m, self.c, self.d)
            rho_new = transport(rho, fr, 0, 1)
            m_new = transport(m, fr, 0, 1)
            gain = np.clip(a1 + a2 * rho + a3 * (m**2).sum(-1) + a4 * rho**2, -1.0, None)
            lap = np.stack([_lap_axes(m[..., k], 0, 1) for k in (0, 1)], -1)
            grad = np.stack([_dx(rho, 0), _dx(rho, 1)], -1)
            m_new = m_new + gain[..., None] * m + a5 * lap + a6 * grad
            if noise:
                r_in = np.clip(rho, *self.rho_range)
                m2_in = np.minimum((m**2).sum(-1), self.m2_max)
                sd = np.exp(0.5 * (self.b[0] + self.b[1] * r_in + self.b[2] * m2_in))
                m_new = m_new + sd[..., None] * rng.standard_normal((g, g, 2))
                fx, fy = self.flux_sd * rng.standard_normal((2, g, g))
                rho_new = rho_new + fx - np.roll(fx, 1, 0) + fy - np.roll(fy, 1, 1)
            rho = np.maximum(rho_new, 1e-3)
            rho *= mass / rho.sum()
            norm = np.sqrt((m_new**2).sum(-1))
            m = m_new * np.minimum(1.0, rho / np.maximum(norm, 1e-12))[..., None]
            phi[t] = np.hypot(*m.sum((0, 1))) / rho.sum()
        return phi[burn:]


def _fit_and_predict(args):
    eta, fields = args
    law = ConservativeFieldLaw(fields)
    det = law.simulate(grid_size(N_FIT), noise=False, start_aligned=True, steps=3000, burn=2999)
    out = {"noise": eta, "a": law.a.tolist(), "c": law.c, "d": law.d,
           "deterministic_phi": float(det[-1])}
    for n in (400, N_FIT, 6400):
        phi = law.simulate(grid_size(n), seed=int(eta * 100))
        out[n] = (float(phi.mean()), float(n * phi.var()))
    return out


def analyze(preds: list[dict], agents: dict, out_dir: Path,
            title: str = "Experiment 18: conservative field law (momentum moves with agents), "
                         "fitted at N=1600",
            name: str = "e18_conservative_field", free_flight: float = 0.4) -> str:
    preds = {p["noise"]: p for p in preds}
    sizes = (400, N_FIT, 6400)
    p30 = preds[3.0]
    lines = [title,
             "", f"  transport fitted on density at noise 3.0: c = {p30['c']:.3f} "
             f"(free-flight value speed*TAU/cell = {free_flight}), D = {p30['d']:.4f}",
             "  local momentum terms (gain a1..a4, Lap m, grad rho) at noise 3.0: "
             + " ".join(f"{a:+.3g}" for a in p30["a"]), "",
             f"  {'noise':>6}{'no-noise phi':>13}" + "".join(f"{f'phi {n} ag.':>13}{'field':>8}"
                                                   for n in sizes)
             + "".join(f"{f'chi {n} ag.':>13}{'field':>8}" for n in (400, 6400))]
    for e in NOISES:
        lines.append(f"  {e:6.2f}{preds[e]['deterministic_phi']:13.3f}"
                     + "".join(f"{agents[n][e][0]:13.3f}{preds[e][n][0]:8.3f}" for n in sizes)
                     + "".join(f"{agents[n][e][1]:13.2f}{preds[e][n][1]:8.2f}"
                               for n in (400, 6400)))
    hits, err, base, peak = {}, {}, {}, {}
    for n in sizes:
        e_field = np.array([abs(preds[e][n][0] - agents[n][e][0]) for e in NOISES])
        hits[n], err[n] = int((e_field <= 0.05).sum()), e_field.sum()
        base[n] = sum(abs(agents[N_FIT][e][0] - agents[n][e][0]) for e in NOISES)
        peak[n] = (float(NOISES[int(np.argmax([preds[e][n][1] for e in NOISES]))]),
                   float(NOISES[int(np.argmax([agents[n][e][1] for e in NOISES]))]))
    p3 = {n: preds[3.0][n][0] for n in sizes}
    s0 = hits[N_FIT] >= 10
    t1 = all(hits[n] >= 10 for n in (400, 6400))
    t2 = p3[400] > p3[N_FIT] > p3[6400]
    t3 = all(err[n] < base[n] for n in (400, 6400))
    t4 = all(abs(peak[n][0] - peak[n][1]) <= 0.25 for n in (400, 6400))
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(s0)}  S0 fitted size N=1600: within 0.05 at {hits[N_FIT]}/13",
              f"  {tag(t1)}  T1 within 0.05 at >= 10/13 "
              + ", ".join(f"[N={n}: {hits[n]}/13]" for n in (400, 6400)),
              f"  {tag(t2)}  T2 order falls with size at noise 3.0 "
              f"[{p3[400]:.3f} > {p3[N_FIT]:.3f} > {p3[6400]:.3f}]",
              f"  {tag(t3)}  T3 beats 'no size dependence' "
              + ", ".join(f"[N={n}: {err[n]:.3f} vs {base[n]:.3f}]" for n in (400, 6400)),
              f"  {tag(t4)}  T4 chi peak within 0.25 "
              + ", ".join(f"[N={n}: {peak[n][0]} vs {peak[n][1]}]" for n in (400, 6400))]
    _plot(preds, agents, out_dir / f"{name}.png")
    lines.append(f"\n  figure: {out_dir / f'{name}.png'}")
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
                       label=f"field law N={n}" + (" (fitted)" if n == N_FIT else ""))
    ax[0].set_title("Global polarization: conservative field law fitted at N=1600")
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
    path = out_dir / "e17_fields_1600.npz"
    if not path.exists():
        raise SystemExit("needs results/e17_fields_1600.npz (run.py field)")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    d = np.load(path)
    jobs = [(float(e), d["fields"][d["noise"] == e]) for e in NOISES]
    print("  fitting and running the conservative field law ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        preds = list(pool.map(_fit_and_predict, jobs))
    (out_dir / "e18_conservative_field_raw.json").write_text(json.dumps(
        [{str(k): v for k, v in p.items()} for p in preds], indent=1))
    return analyze(preds, _agents(), out_dir)


if __name__ == "__main__":
    main()
