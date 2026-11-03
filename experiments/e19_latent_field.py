"""Experiment 19: hidden-state field law (latent field + known counting noise).

Experiments 17 and 18 fitted a field law directly on cell averages of ~25
agents. Those averages are noisy readings of the local flow, so the fitted
"random kicks" absorbed the counting noise, and least squares was biased
towards decay; the iterated law never ordered.

Model:
    hidden field  z_t = (rho_t, m_t) on the lattice, evolving by the
                  conservative law of experiment 18 plus process noise
                  (sd sigma_m per momentum component, conservative density
                  fluxes with sd sigma_f);
    observation   y_t = z_t + e_t, e_t ~ N(0, R), with the counting noise of a
                  compound Poisson cell of N0 = 25 expected agents:
                  Var(rho) = rho / N0,  Var(m_k) = rho / (2 N0)  (known, not fitted).

Fitting (per noise level, N = 1600 data of experiment 17), EM_ITERS rounds:
    E-step  ensemble transform Kalman filter with a fixed-lag smoother
            (ENSEMBLE members, lag LAG) gives the hidden field's mean path;
    M-step  transport (c, D >= 0) and local momentum coefficients refitted by
            least squares on the smoothed path; sigma_m and sigma_f rescaled so
            that the filter's innovations have the variance the model expects.

Prediction: the fitted hidden law is run freely on 4x4, 8x8 and 16x16 lattices;
counting noise is added back and global phi = |sum m_obs| / sum rho_obs is
computed exactly as for the agents.

STATUS: the estimator failed validation on synthetic data generated from a
known hidden law with exactly this observation model (8x8, 3 runs x 800
steps, true c = 0.33, sigma_m = 0.05). From either a least-squares start or a
small-noise start, EM converged to c ~ 0.13, sigma_m ~ 0.35, and a law that
does not order (phi 0.10 vs 0.52): with 25-agent cells the counting noise
swamps the one-step signal, the M-step on smoothed means inherits the bad
start, and the innovation check lets process noise absorb model error. The
agent test below was therefore NOT run; its predictions remain untested.

Predictions, fixed before running (identical to experiments 17 and 18):

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
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e11_band_closure import NOISES
from experiments.e17_field_law import BURN, N0, N_FIT, STEPS, _agents, _dx, _lap_axes, grid_size
from experiments.e18_conservative_field import _fractions, _transport_parts, local_features, transport

ENSEMBLE = 64
LAG = 4
EM_ITERS = 6
INFLATION = 1.02
SPIN_UP = 10


class Law:
    def __init__(self, c, d, a, sigma_m, sigma_f):
        self.c, self.d, self.a = float(c), float(max(d, 0.0)), np.asarray(a, float)
        self.sigma_m, self.sigma_f = float(sigma_m), float(sigma_f)

    def step(self, rho, m, rng, noise=True):
        """Batched over leading axes; lattice axes are the last two of rho."""
        ax0, ax1 = rho.ndim - 2, rho.ndim - 1
        fr = _fractions(rho, m, self.c, self.d)
        rho_n, m_n = transport(rho, fr, ax0, ax1), transport(m, fr, ax0, ax1)
        a1, a2, a3, a4, a5, a6 = self.a
        gain = np.clip(a1 + a2 * rho + a3 * (m**2).sum(-1) + a4 * rho**2, -1.0, None)
        lap = np.stack([_lap_axes(m[..., k], ax0, ax1) for k in (0, 1)], -1)
        grad = np.stack([_dx(rho, ax0), _dx(rho, ax1)], -1)
        m_n = m_n + gain[..., None] * m + a5 * lap + a6 * grad
        if noise:
            m_n = m_n + self.sigma_m * rng.standard_normal(m_n.shape)
            fx, fy = self.sigma_f * rng.standard_normal((2, *rho.shape))
            rho_n = rho_n + fx - np.roll(fx, 1, ax0) + fy - np.roll(fy, 1, ax1)
        rho_n = np.maximum(rho_n, 1e-3)
        norm = np.sqrt((m_n**2).sum(-1))
        m_n = m_n * np.minimum(1.0, rho_n / np.maximum(norm, 1e-12))[..., None]
        return rho_n, m_n

    def free_run(self, g, seed=0, noise=True, start_aligned=False, steps=STEPS, burn=BURN):
        rng = np.random.default_rng(seed)
        rho = np.ones((g, g))
        ang = np.zeros((g, g)) if start_aligned else rng.uniform(-np.pi, np.pi, (g, g))
        m = (0.9 if start_aligned else 0.3) * np.stack([np.cos(ang), np.sin(ang)], -1)
        mass = rho.sum()
        phi = np.empty(steps)
        for t in range(steps):
            rho, m = self.step(rho, m, rng, noise)
            rho *= mass / rho.sum()
            if noise:
                sd = np.sqrt(rho / N0)
                rho_o = rho + sd * rng.standard_normal(rho.shape)
                m_o = m + (sd / np.sqrt(2))[..., None] * rng.standard_normal(m.shape)
                phi[t] = np.hypot(*m_o.sum((0, 1))) / rho_o.sum()
            else:
                phi[t] = np.hypot(*m.sum((0, 1))) / rho.sum()
        return phi[burn:]


def fit_law(rho, m) -> Law:
    """Least-squares M-step on (runs, T, g, g) paths."""
    ax0, ax1 = rho.ndim - 2, rho.ndim - 1
    r0, r1, m0, m1 = rho[:, :-1], rho[:, 1:], m[:, :-1], m[:, 1:]
    ar, br = _transport_parts(r0, r0, m0, ax0, ax1)
    y = (r1 - r0).reshape(-1)
    c, d = np.linalg.lstsq(np.stack([ar.reshape(-1), br.reshape(-1)], 1), y, rcond=None)[0]
    if d < 0:
        d = 0.0
        c = float(ar.reshape(-1) @ y / (ar.reshape(-1) @ ar.reshape(-1)))
    res_r = y - (c * ar + d * br).reshape(-1)
    moved = transport(m0, _fractions(r0, m0, c, d), ax0, ax1)
    x = np.stack([t.reshape(-1) for t in local_features(r0, m0, ax0, ax1)], 1)
    target = (m1 - moved).reshape(-1)
    a = np.linalg.lstsq(x, target, rcond=None)[0]
    res_m = target - x @ a
    return Law(c, d, a, res_m.std(), np.sqrt(res_r.var() / 4))


def _obs_var(y):
    r = np.maximum(y[..., 0], 0.2) / N0
    return np.stack([r, r / 2, r / 2], -1)


def smooth(law: Law, obs: np.ndarray, seed: int = 0):
    """ETKF with fixed-lag smoother. obs: (T, g, g, 3). Returns smoothed means and
    innovation ratios (observed / expected innovation variance) for rho and m."""
    rng = np.random.default_rng(seed)
    t_len, g = obs.shape[0], obs.shape[1]
    sd0 = np.sqrt(_obs_var(obs[0]))
    ens = obs[0][None] + sd0[None] * rng.standard_normal((ENSEMBLE, g, g, 3))
    ens[..., 0] = np.maximum(ens[..., 0], 1e-3)
    lagged: deque = deque()
    out = np.empty_like(obs, dtype=float)
    num, den = np.zeros(2), np.zeros(2)
    sq = np.sqrt(ENSEMBLE - 1)
    for t in range(t_len):
        if t > 0:
            rho, m = law.step(ens[..., 0], ens[..., 1:], rng)
            ens = np.concatenate([rho[..., None], m], -1)
            mean = ens.mean(0)
            ens = mean + INFLATION * (ens - mean)
        x = ens.reshape(ENSEMBLE, -1)
        xm = x.mean(0)
        anom = (x - xm) / sq
        r = _obs_var(obs[t]).reshape(-1)
        innov = obs[t].reshape(-1) - xm
        spread = (anom**2).sum(0)
        for k, sl in enumerate((slice(0, None, 3), None)):
            mask = np.zeros(innov.size, bool)
            if sl is None:
                mask[1::3], mask[2::3] = True, True
            else:
                mask[sl] = True
            num[k] += (innov[mask] ** 2).sum()
            den[k] += (spread[mask] + r[mask]).sum()
        s = anom / np.sqrt(r)
        lam, v = np.linalg.eigh(np.eye(ENSEMBLE) + s @ s.T)
        wbar = v @ ((v.T @ (s @ (innov / np.sqrt(r)))) / lam)
        w = v @ np.diag(lam**-0.5) @ v.T

        def update(e):
            flat = e.reshape(ENSEMBLE, -1)
            mu = flat.mean(0)
            a = (flat - mu) / sq
            new = mu + wbar @ a + sq * (w @ a)
            return new.reshape(e.shape)

        ens = update(ens)
        ens[..., 0] = np.maximum(ens[..., 0], 1e-3)
        lagged = deque(update(e) for e in lagged)
        lagged.append(ens.copy())
        if len(lagged) > LAG:
            out[t - LAG] = lagged.popleft().mean(0)
    for k, e in enumerate(lagged):
        out[t_len - len(lagged) + k] = e.mean(0)
    return out, num / den


def em_fit(fields: np.ndarray, log: list | None = None) -> Law:
    f = fields.astype(float)
    law = fit_law(f[..., 0], f[..., 1:])
    for it in range(EM_ITERS):
        paths, ratios = zip(*(smooth(law, run, seed=i) for i, run in enumerate(f)))
        paths = np.array(paths)[:, SPIN_UP:]
        ratio = np.mean(ratios, 0)
        new = fit_law(paths[..., 0], paths[..., 1:])
        new.sigma_m = law.sigma_m * np.sqrt(np.clip(ratio[1], 0.25, 4.0))
        new.sigma_f = law.sigma_f * np.sqrt(np.clip(ratio[0], 0.25, 4.0))
        law = new
        if log is not None:
            log.append({"iter": it, "c": law.c, "d": law.d, "a": law.a.tolist(),
                        "sigma_m": law.sigma_m, "sigma_f": law.sigma_f,
                        "innovation_ratio": ratio.tolist()})
    return law


def _fit_and_predict(args):
    eta, fields = args
    log: list = []
    law = em_fit(fields, log)
    det = law.free_run(grid_size(N_FIT), noise=False, start_aligned=True, steps=3000, burn=2999)
    out = {"noise": eta, "em": log, "deterministic_phi": float(det[-1])}
    for n in (400, N_FIT, 6400):
        phi = law.free_run(grid_size(n), seed=int(eta * 100))
        out[n] = (float(phi.mean()), float(n * phi.var()))
    return out


def analyze(preds: list[dict], agents: dict, out_dir: Path) -> str:
    preds = {p["noise"]: p for p in preds}
    sizes = (400, N_FIT, 6400)
    last = preds[3.0]["em"][-1]
    lines = ["Experiment 19: hidden-state field law (EM + ensemble Kalman smoother), "
             "fitted at N=1600", "",
             f"  noise 3.0 after EM: c = {last['c']:.3f}, D = {last['d']:.4f}, "
             f"local a = " + " ".join(f"{a:+.3g}" for a in last["a"])
             + f", sigma_m = {last['sigma_m']:.3f}, sigma_f = {last['sigma_f']:.3f}",
             "  innovation ratio (rho, m) per EM iteration at noise 3.0: "
             + "  ".join(f"({r['innovation_ratio'][0]:.2f}, {r['innovation_ratio'][1]:.2f})"
                         for r in preds[3.0]["em"]), "",
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
    _plot(preds, agents, out_dir / "e19_latent_field.png")
    lines.append(f"\n  figure: {out_dir / 'e19_latent_field.png'}")
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
                       label=f"hidden-field law N={n}" + (" (fitted)" if n == N_FIT else ""))
    ax[0].set_title("Global polarization: hidden-state field law fitted at N=1600")
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
    print("  EM fitting of the hidden-state field law ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        preds = list(pool.map(_fit_and_predict, jobs))
    (out_dir / "e19_latent_field_raw.json").write_text(json.dumps(
        [{str(k): v for k, v in p.items()} for p in preds], indent=1))
    return analyze(preds, _agents(), out_dir)


if __name__ == "__main__":
    main()
