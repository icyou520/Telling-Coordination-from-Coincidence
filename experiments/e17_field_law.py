"""Experiment 17: a coarse-grained field law for the flock, fitted at one size.

Experiment 16 found a size-independent layer at a fixed scale (regional and
local order), but global order and the transition, which depend on size, live
in how regions align with each other. Here the next layer up is modelled as a
field on a lattice of cells of side ELL = 2.5 (25 agents per cell on average):

    rho_i  = agents in cell i / 25                     (density)
    m_i    = sum of heading vectors in cell i / 25     (momentum)

One macro step = TAU = 10 agent steps (agents move 1 unit = 0.4 cells). The
field law is local, fitted separately at each noise level on N = 1600 only
(an 8 x 8 lattice), with the standard flocking-hydrodynamics terms:

    dm  = a1 m + a2 rho m + a3 |m|^2 m + a4 rho^2 m          (alignment, saturation)
        + a5 Lap m + a6 grad rho + a7 (m.grad) m             (diffusion, pressure,
        + a8 m (div m) + a9 grad |m|^2                        advection)
        + noise, variance exp(b0 + b1 rho + b2 |m|^2) per component
    drho = c (upwind transport of agents along m) + D Lap rho + conservative random fluxes

Numerical guards (fixed before any test-size comparison): total mass is held
exactly constant, the local gain a1 + a2 rho + a3 |m|^2 + a4 rho^2 cannot drop
below -1 (decay cannot flip m), |m| <= rho, and the noise model is evaluated
inside the training range of rho and |m|^2.

The same law is then run on 4 x 4 (N = 400) and 16 x 16 (N = 6400) lattices.
Nothing about size enters the law; any size dependence of global order
phi = |sum m| / sum rho has to come out of the field dynamics.

Predictions, fixed before running (agent values from experiment 16):

  T1  Mean global phi within 0.05 of the agents at >= 10/13 noise levels, at
      both N = 400 and N = 6400.
  T2  At noise 3.0 the predicted order falls with size:
      phi(400) > phi(1600) > phi(6400)   (agents: 0.551 > 0.461 > 0.345).
  T3  Smaller summed |error| of mean phi than assuming no size dependence
      (using the N = 1600 agent curve), at both sizes.
  T4  Susceptibility N Var(phi) peaks within 0.25 of the agents' peak, at both
      sizes (agents: 3.75 at N = 400, 3.0 at N = 6400).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e9_interaction_closure import BURN_IN, N_MACRO, TAU
from experiments.e11_band_closure import NOISES

ELL = 2.5
N0 = 4.0 * ELL**2
N_FIT = 1600
SEEDS = range(800, 804)
STEPS, BURN = 20_000, 2_000


def grid_size(n: int) -> int:
    return int(round(np.sqrt(n / 4.0) / ELL))


def cell_fields(s: np.ndarray, box: float, g: int):
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
    rng = np.random.default_rng(70_000 * n + 101 * seed + int(eta * 1000))
    s = model.initial(rng)
    for _ in range(BURN_IN):
        s = model.step(s, rng)
    out = np.empty((N_MACRO, g, g, 3), np.float32)
    for t in range(N_MACRO):
        for _ in range(TAU):
            s = model.step(s, rng)
        out[t] = cell_fields(s, model.box, g)
    return float(eta), seed, out


def _dx(f, axis):
    return 0.5 * (np.roll(f, -1, axis) - np.roll(f, 1, axis))


def momentum_features(rho, m):
    """Rotation/reflection-equivariant local terms, each shaped like m (..., g, g, 2)."""
    ax0, ax1 = rho.ndim - 2, rho.ndim - 1
    m2 = (m**2).sum(-1)
    grad = lambda f: np.stack([_dx(f, ax0), _dx(f, ax1)], -1)
    div = _dx(m[..., 0], ax0) + _dx(m[..., 1], ax1)
    adv = np.stack([m[..., 0] * _dx(m[..., k], ax0) + m[..., 1] * _dx(m[..., k], ax1)
                    for k in (0, 1)], -1)
    lap = np.stack([_lap_axes(m[..., k], ax0, ax1) for k in (0, 1)], -1)
    return [m, rho[..., None] * m, m2[..., None] * m, rho[..., None] ** 2 * m,
            lap, grad(rho), adv, m * div[..., None], grad(m2)]


def _lap_axes(f, ax0, ax1):
    return sum(np.roll(f, k, a) for a in (ax0, ax1) for k in (-1, 1)) - 4 * f


def density_features(rho, m):
    """Upwind transport (each cell sends agents along its momentum) and diffusion."""
    ax0, ax1 = rho.ndim - 2, rho.ndim - 1
    transport = 0.0
    for k, ax in ((0, ax0), (1, ax1)):
        fwd, bwd = np.maximum(m[..., k], 0), np.maximum(-m[..., k], 0)
        transport = transport + np.roll(fwd, 1, ax) + np.roll(bwd, -1, ax) - fwd - bwd
    return [transport, _lap_axes(rho, ax0, ax1)]


class FieldLaw:
    def __init__(self, fields: np.ndarray):
        """fields: (runs, T, g, g, 3) at one noise level."""
        f = fields.astype(float)
        rho, m = f[:, :-1, ..., 0], f[:, :-1, ..., 1:]
        d_rho, d_m = np.diff(f[..., 0], axis=1), np.diff(f[..., 1:], axis=1)
        xm = np.stack([t.reshape(-1) for t in momentum_features(rho, m)], 1)
        self.a = np.linalg.lstsq(xm, d_m.reshape(-1), rcond=None)[0]
        res = (d_m.reshape(-1) - xm @ self.a).reshape(d_m.shape)
        m2 = (m**2).sum(-1)
        xs = np.stack([np.ones(rho.size), rho.reshape(-1), m2.reshape(-1)], 1)
        r2 = (res**2).mean(-1).reshape(-1)
        self.b = np.linalg.lstsq(xs, np.log(r2 + 1e-12), rcond=None)[0]
        self.b[0] += np.log(np.mean(r2 / np.exp(xs @ self.b)))
        xr = np.stack([t.reshape(-1) for t in density_features(rho, m)], 1)
        self.c = np.linalg.lstsq(xr, d_rho.reshape(-1), rcond=None)[0]
        self.flux_sd = float(np.sqrt(np.var(d_rho.reshape(-1) - xr @ self.c) / 4))
        self.rho_range = (float(rho.min()), float(rho.max()))
        self.m2_max = float(m2.max())

    def simulate(self, g: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        rho = np.ones((g, g))
        ang = rng.uniform(-np.pi, np.pi, (g, g))
        m = 0.3 * np.stack([np.cos(ang), np.sin(ang)], -1)
        phi = np.empty(STEPS)
        mass = rho.sum()
        for t in range(STEPS):
            feats = momentum_features(rho, m)
            a1, a2, a3, a4 = self.a[:4]
            gain = a1 + a2 * rho + a3 * (m**2).sum(-1) + a4 * rho**2
            gain = np.clip(gain, -1.0, None)[..., None]
            dm = gain * m + sum(a * f for a, f in zip(self.a[4:], feats[4:]))
            r_in = np.clip(rho, *self.rho_range)
            m2_in = np.minimum((m**2).sum(-1), self.m2_max)
            sd = np.exp(0.5 * (self.b[0] + self.b[1] * r_in + self.b[2] * m2_in))
            m = m + dm + sd[..., None] * rng.standard_normal((g, g, 2))
            dr = sum(c * f for c, f in zip(self.c, density_features(rho, m)))
            fx, fy = self.flux_sd * rng.standard_normal((2, g, g))
            rho = rho + dr + fx - np.roll(fx, 1, 0) + fy - np.roll(fy, 1, 1)
            rho = np.maximum(rho, 1e-3)
            rho *= mass / rho.sum()
            norm = np.sqrt((m**2).sum(-1))
            m = m * np.minimum(1.0, rho / np.maximum(norm, 1e-12))[..., None]
            phi[t] = np.hypot(*m.sum((0, 1))) / rho.sum()
        return phi[BURN:]


def _fit_and_predict(args):
    eta, fields = args
    law = FieldLaw(fields)
    out = {"noise": eta, "a": law.a.tolist(), "c": law.c.tolist()}
    for n in (400, N_FIT, 6400):
        phi = law.simulate(grid_size(n), seed=int(eta * 100))
        out[n] = (float(phi.mean()), float(n * phi.var()))
    return out


def _agents() -> dict:
    d = np.load(Path("results") / "e16_trajectories.npz")
    return {n: {float(e): (float(d["phi"][(d["n"] == n) & (d["noise"] == e)].mean()),
                           float(np.mean([n * p.var() for p in
                                          d["phi"][(d["n"] == n) & (d["noise"] == e)]])))
                for e in NOISES} for n in (400, N_FIT, 6400)}


def analyze(preds: list[dict], agents: dict, out_dir: Path) -> str:
    preds = {p["noise"]: p for p in preds}
    lines = ["Experiment 17: field law fitted at N=1600, run on 4x4 / 8x8 / 16x16 lattices", "",
             "  fitted momentum coefficients (alignment ... advection) at noise 3.0:",
             "    " + " ".join(f"{a:+.3g}" for a in preds[3.0]["a"]),
             f"    density: c transport + D Lap rho with c = {preds[3.0]['c'][0]:.3g}, "
             f"D = {preds[3.0]['c'][1]:.3g}", "",
             f"  {'noise':>6}" + "".join(f"{f'phi {n} ag.':>13}{f'field':>8}"
                                         for n in (400, N_FIT, 6400))
             + "".join(f"{f'chi {n} ag.':>13}{f'field':>8}" for n in (400, 6400))]
    for e in NOISES:
        lines.append(f"  {e:6.2f}" + "".join(f"{agents[n][e][0]:13.3f}{preds[e][n][0]:8.3f}"
                                             for n in (400, N_FIT, 6400))
                     + "".join(f"{agents[n][e][1]:13.2f}{preds[e][n][1]:8.2f}"
                               for n in (400, 6400)))
    hits, err, base, peak = {}, {}, {}, {}
    for n in (400, 6400):
        e_field = np.array([abs(preds[e][n][0] - agents[n][e][0]) for e in NOISES])
        e_base = np.array([abs(agents[N_FIT][e][0] - agents[n][e][0]) for e in NOISES])
        hits[n], err[n], base[n] = int((e_field <= 0.05).sum()), e_field.sum(), e_base.sum()
        peak[n] = (float(NOISES[int(np.argmax([preds[e][n][1] for e in NOISES]))]),
                   float(NOISES[int(np.argmax([agents[n][e][1] for e in NOISES]))]))
    p3 = {n: preds[3.0][n][0] for n in (400, N_FIT, 6400)}
    t1 = all(h >= 10 for h in hits.values())
    t2 = p3[400] > p3[N_FIT] > p3[6400]
    t3 = all(err[n] < base[n] for n in (400, 6400))
    t4 = all(abs(a - b) <= 0.25 for a, b in peak.values())
    in_sample = int(sum(abs(preds[e][N_FIT][0] - agents[N_FIT][e][0]) <= 0.05 for e in NOISES))
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", f"  sanity (fitted size N=1600): mean phi within 0.05 at {in_sample}/13",
              "", "  Pre-registered predictions:",
              f"  {tag(t1)}  T1 mean phi within 0.05 at >= 10/13 "
              + ", ".join(f"[N={n}: {h}/13]" for n, h in hits.items()),
              f"  {tag(t2)}  T2 order falls with size at noise 3.0 "
              f"[{p3[400]:.3f} > {p3[N_FIT]:.3f} > {p3[6400]:.3f}]",
              f"  {tag(t3)}  T3 beats 'no size dependence' "
              + ", ".join(f"[N={n}: {err[n]:.3f} vs {base[n]:.3f}]" for n in (400, 6400)),
              f"  {tag(t4)}  T4 chi peak within 0.25 "
              + ", ".join(f"[N={n}: {a} vs {b}]" for n, (a, b) in peak.items())]
    _plot(preds, agents, out_dir / "e17_field_law.png")
    lines.append(f"\n  figure: {out_dir / 'e17_field_law.png'}")
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
    ax[0].set_title("Global polarization: field law fitted at N=1600 only")
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
    if not (out_dir / "e16_trajectories.npz").exists():
        raise SystemExit("needs results/e16_trajectories.npz (run.py regional)")
    path = out_dir / "e17_fields_1600.npz"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    workers = max(1, (os.cpu_count() or 2) - 2)
    if not path.exists():
        seeds = list(SEEDS)[:1] if quick else list(SEEDS)
        tasks = [(N_FIT, e, s) for e in NOISES for s in seeds]
        print(f"  simulating {len(tasks)} runs at N={N_FIT} ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            sims = list(pool.map(_run, tasks, chunksize=1))
        np.savez_compressed(path, noise=np.array([s[0] for s in sims]),
                            fields=np.array([s[2] for s in sims]))
    d = np.load(path)
    jobs = [(float(e), d["fields"][d["noise"] == e]) for e in NOISES]
    print("  fitting and running the field law ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        preds = list(pool.map(_fit_and_predict, jobs))
    (out_dir / "e17_field_law_raw.json").write_text(json.dumps(
        [{k if isinstance(k, str) else str(k): v for k, v in p.items()} for p in preds], indent=1))
    return analyze(preds, _agents(), out_dir)


if __name__ == "__main__":
    main()
