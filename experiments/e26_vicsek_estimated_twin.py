"""Experiment 26: the no-interaction twin of a flock, estimated from data.

e9 used the exact twin law: headings of non-interacting agents only pick up
uniform noise of width eta, so E[cos(kick)] = c = sin(eta/2)/(eta/2) and
|V'| is Rice distributed. Here c is instead estimated from agent data, as an
analyst without access to eta or to a twin simulation would have to.

Mechanistic fit (as in e24: own state plus relative state). From pairs of
consecutive snapshots, with psi_i the mean heading of the agents within an
assumed radius R (self included),

    theta_i' = theta_i + kappa * wrap(psi_i - theta_i) + xi,

kappa is chosen to maximize mean cos(xi), and c_hat = mean cos(xi). The twin
keeps only xi. Naive twin: c_naive = mean cos(theta_i' - theta_i).

S is computed exactly as in e9 (cubic G on 70 % of the series, scored on the
rest) with c_hat in the Rice aggregation law, on the same runs as the exact
twin, so differences come only from the estimated c.

Predictions, fixed before running:

  V1  Fitted twin with the true radius (R = 1): |S - S_exact| <= 0.05 at
      every noise, for flocks and for non-interacting agents.
  V2  Non-interacting agents: S <= 0.1 at every noise with every estimated
      twin (R = 0.5, 1, 2 and naive): no false positives without interaction.
  V3  Flocks with a misspecified radius (R = 0.5, 2): |S - S_exact| <= 0.1 at
      every noise.

Added after the first run, before the second: V1-V3 passed, but S was
identical to three decimals for every twin, because at tau = 10 the twin's
prediction depends on c^10, which is negligible or swamped. The test was
therefore not discriminating. A one-step horizon (tau = 1, 2000 extra
consecutive micro steps per run) makes S depend on c itself:

  V4  tau = 1, true radius: |S - S_exact| <= 0.05 at every noise, flocks and
      non-interacting agents.
  V5  tau = 1, naive twin: |S - S_exact| > 0.05 at some noise for flocks
      (its biased c matters once the horizon is short).
  V6  tau = 1, non-interacting agents: S <= 0.1 for every estimated twin.
Exploratory: tau = 1 with a misspecified radius.

The first tau = 1 run returned NaN for ordered flocks: scipy's Rice mean
overflows at signal-to-noise ratios above ~30. The large-ratio expansion
nu + sigma^2 / (2 nu) (relative error < 1e-6 there) is used instead.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e9_interaction_closure import BURN_IN, N_AGENTS, N_MACRO, NOISES, TAU

RADII = (0.5, 1.0, 2.0)
SEEDS = 2
PAIRS = 200
KAPPA_GRID = np.linspace(0.0, 1.5, 151)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fit_c(theta, theta_next, psi) -> tuple[float, float]:
    """(kappa, c_hat) maximizing mean cos of the residual heading kick."""
    d, r = _wrap(theta_next - theta), _wrap(psi - theta)
    scores = np.array([np.cos(d - k * r).mean() for k in KAPPA_GRID])
    i = int(np.argmax(scores))
    return float(KAPPA_GRID[i]), float(scores[i])


def _run(args):
    eta, seed, kind = args
    from emergence_lab.systems.vicsek import Vicsek

    model = Vicsek(n_agents=N_AGENTS, noise=float(eta), radius=1.0 if kind == "flock" else 0.0)
    probes = {r: Vicsek(n_agents=N_AGENTS, radius=r) for r in RADII}
    rng = np.random.default_rng(260_000 + 101 * seed + int(eta * 1000) + (7 if kind == "twin" else 0))
    s = model.initial(rng)
    for _ in range(BURN_IN):
        s = model.step(s, rng)
    phi = np.empty(N_MACRO)
    every = N_MACRO // PAIRS
    th, th_next, psi = [], [], {r: [] for r in RADII}
    for t in range(N_MACRO):
        for k in range(TAU):
            if k == 0 and t % every == 0:
                for r, p in probes.items():
                    ss, cc = p.neighbour_sums(s[:, :2], s[:, 2])
                    psi[r].append(np.arctan2(ss, cc))
                th.append(s[:, 2].copy())
                s = model.step(s, rng)
                th_next.append(s[:, 2].copy())
            else:
                s = model.step(s, rng)
        phi[t] = np.hypot(np.cos(s[:, 2]).mean(), np.sin(s[:, 2]).mean())
    phi1 = np.empty(N_MACRO)
    for t in range(N_MACRO):
        s = model.step(s, rng)
        phi1[t] = np.hypot(np.cos(s[:, 2]).mean(), np.sin(s[:, 2]).mean())
    th, th_next = np.concatenate(th), np.concatenate(th_next)
    fits = {f"R={r}": fit_c(th, th_next, np.concatenate(psi[r])) for r in RADII}
    fits["naive"] = (0.0, float(np.cos(_wrap(th_next - th)).mean()))
    c_true = float(np.sin(eta / 2) / (eta / 2))
    c_hats = {k: v[1] for k, v in fits.items()}
    one = score(phi1, c_true, c_hats, tau=1)
    return {"noise": float(eta), "seed": seed, "kind": kind, "c_true": c_true,
            "fits": fits, **score(phi, c_true, c_hats),
            "S_exact_1": one["S_exact"], "S_1": one["S"]}


def _rice_mean(phi, c, n, tau=TAU):
    from scipy.stats import rice

    nu = np.clip(c, 0.0, 1.0) ** tau * phi
    sigma = np.sqrt(max(1 - np.clip(c, 0.0, 1.0) ** (2 * tau), 1e-12) / (2 * n))
    snr = nu / sigma
    large = snr > 30.0
    out = np.empty_like(nu)
    out[~large] = rice.mean(snr[~large], scale=sigma)
    out[large] = nu[large] + sigma ** 2 / (2 * nu[large])
    return out


def score(phi, c_true, c_hats, tau=TAU) -> dict:
    from emergence_lab.closure import fit_held_out
    from emergence_lab.layers import poly

    z, zn = phi[:-1, None], phi[1:, None]
    cut = int(len(z) * 0.7)
    tr, te = np.arange(cut), np.arange(cut, len(z))
    _, pred = fit_held_out(poly(z, 3), zn, tr, te, np.random.default_rng(0))
    mse_g = float(((zn[te] - pred) ** 2).mean())
    share = lambda c: 1 - mse_g / float(
        ((zn[te, 0] - _rice_mean(z[te, 0], c, N_AGENTS, tau)) ** 2).mean())
    return {"S_exact": share(c_true), "S": {k: share(c) for k, c in c_hats.items()}}


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    seeds = 1 if quick else SEEDS
    tasks = [(float(e), s, k) for k in ("flock", "twin") for e in NOISES for s in range(seeds)]
    print(f"  running {len(tasks)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(_run, tasks, chunksize=1))
    (out_dir / "e26_vicsek_twin_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    keys = [f"R={r}" for r in RADII] + ["naive"]
    mean = lambda kind, e, f: float(np.mean([f(r) for r in rows
                                             if r["kind"] == kind and r["noise"] == e]))
    lines = [f"Experiment 26: estimated twin for Vicsek flocks (N={N_AGENTS}, tau={TAU})", ""]
    for kind in ("flock", "twin"):
        lines.append(f"  {kind}: S with each twin (c estimate in brackets; true c first)")
        lines.append(f"  {'noise':>6}{'exact':>16}" + "".join(f"{k:>17}" for k in keys))
        for e in NOISES:
            cells = f"{mean(kind, e, lambda r: r['S_exact']):7.3f} [{mean(kind, e, lambda r: r['c_true']):.3f}]"
            for k in keys:
                cells += (f"{mean(kind, e, lambda r: r['S'][k]):8.3f} "
                          f"[{mean(kind, e, lambda r: r['fits'][k][1]):.3f}]")
            lines.append(f"  {e:6.2f}  {cells}")
        lines.append(f"  kappa fitted (R=1): " + ", ".join(
            f"{mean(kind, e, lambda r: r['fits']['R=1.0'][0]):.2f}" for e in NOISES))
        lines.append("")

    dev = lambda kind, k: max(abs(mean(kind, e, lambda r: r["S"][k] - r["S_exact"])) for e in NOISES)
    v1_err = max(dev("flock", "R=1.0"), dev("twin", "R=1.0"))
    v1 = v1_err <= 0.05
    v2_max = max(mean("twin", e, lambda r: r["S"][k]) for e in NOISES for k in keys)
    v2 = v2_max <= 0.1
    v3_err = max(dev("flock", "R=0.5"), dev("flock", "R=2.0"))
    v3 = v3_err <= 0.1
    lines.append("  tau = 1: S (exact twin | R=0.5 | R=1 | R=2 | naive)")
    for kind in ("flock", "twin"):
        for e in NOISES:
            vals = [mean(kind, e, lambda r: r["S_exact_1"])] + [
                mean(kind, e, lambda r: r["S_1"][k]) for k in keys]
            lines.append(f"  {kind:>6} {e:5.2f}  " + "  ".join(f"{v:7.3f}" for v in vals))
    lines.append("")
    dev1 = lambda kind, k: max(abs(mean(kind, e, lambda r: r["S_1"][k] - r["S_exact_1"]))
                               for e in NOISES)
    v4_err = max(dev1("flock", "R=1.0"), dev1("twin", "R=1.0"))
    v4 = v4_err <= 0.05
    v5_dev = dev1("flock", "naive")
    v5 = v5_dev > 0.05
    v6_max = max(mean("twin", e, lambda r: r["S_1"][k]) for e in NOISES for k in keys)
    v6 = v6_max <= 0.1
    v_mis = max(dev1("flock", "R=0.5"), dev1("flock", "R=2.0"))

    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["  Pre-registered predictions:",
              f"  {tag(v1)}  V1 true-radius twin within 0.05 of exact [max deviation {v1_err:.3f}]",
              f"  {tag(v2)}  V2 no false positives without interaction [max S {v2_max:.3f}]",
              f"  {tag(v3)}  V3 misspecified radius within 0.1 for flocks [max deviation {v3_err:.3f}]",
              f"  (naive twin, flocks: max deviation {dev('flock', 'naive'):.3f})",
              f"  {tag(v4)}  V4 tau=1, true radius within 0.05 [max deviation {v4_err:.3f}]",
              f"  {tag(v5)}  V5 tau=1, naive twin deviates > 0.05 for flocks "
              f"[max deviation {v5_dev:.3f}]",
              f"  {tag(v6)}  V6 tau=1, no false positives without interaction [max S {v6_max:.3f}]",
              f"  (exploratory: tau=1, misspecified radius, flocks: max deviation {v_mis:.3f})"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
