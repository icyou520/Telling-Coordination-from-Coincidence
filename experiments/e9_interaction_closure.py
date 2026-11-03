"""Experiment 9: is the flock's macro law made by interactions, and is it closed?

Experiment 8 showed that Psi > 0 is produced by non-interacting agents, so it
cannot detect interaction-driven emergence. Closure alone cannot either: for
agents that ignore each other, the next mean velocity depends on the micro
state only through the current mean (independent noise averages out), so the
aggregate is closed by construction, like an ideal gas.

Two separate questions are therefore asked of the macro variable
polarization phi = |mean heading|, observed every TAU micro steps:

  Interaction share  S = 1 - MSE(fitted macro law G) / MSE(aggregation law G0)
      G0 is the law for independent agents with the same noise: V' = c^TAU V
      plus Gaussian noise, c = sin(eta/2)/(eta/2), so phi' is Rice distributed
      (see aggregation_law).
      S ~ 0: the macro law is just averaging. S ~ 1: interactions make it.

  Autonomy   share of the predictable macro change captured by phi alone,
      versus phi plus physically meaningful micro detail: distribution of
      local alignment, crowding, and density banding along x and y.

Each flock run has a non-interacting twin (alignment radius 0, same noise).

Predictions, fixed before running:

  E1  Twin: S <= 0.1 at every noise, and autonomy >= 0.9 wherever defined.
  E2  Flock: S >= 0.5 at every noise below the susceptibility peak.
  E3  Flock: autonomy at the susceptibility-peak noise is lower than at the
      lowest noise by at least 0.1 (undefined autonomy counts as failure).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

N_AGENTS = 800
TAU = 10
N_MACRO, BURN_IN = 2000, 4000
NOISES = np.round(np.arange(1.0, 5.51, 0.5), 3)
SEEDS = 3
BLOCK = 100


def micro_features(s: np.ndarray, box: float) -> np.ndarray:
    """Rotation-invariant micro detail: local alignment, crowding, density bands."""
    from scipy.spatial import cKDTree

    pos, th = np.where(s[:, :2] >= box, 0.0, s[:, :2]), s[:, 2]
    n = len(s)
    i, j = cKDTree(pos, boxsize=box).query_pairs(1.0, output_type="ndarray").T
    count = 1 + np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    sn, cs = np.sin(th), np.cos(th)
    s_sum = sn + np.bincount(i, sn[j], n) + np.bincount(j, sn[i], n)
    c_sum = cs + np.bincount(i, cs[j], n) + np.bincount(j, cs[i], n)
    local_order = np.hypot(s_sum, c_sum) / count
    order_hist = np.histogram(local_order, bins=8, range=(0, 1))[0] / n
    crowd_hist = np.histogram(count, bins=[1, 2, 3, 5, 8, 13, 21, 34, 10**6])[0] / n
    bands = []
    for ax in (0, 1):
        f = np.abs(np.fft.rfft(np.histogram(pos[:, ax], bins=32, range=(0, box))[0])) / n
        bands += [f[1], f[2]]
    return np.concatenate([order_hist, crowd_hist, bands])


def _run(args):
    n, eta, seed, kind = args
    from emergence_lab.systems.vicsek import Vicsek

    model = Vicsek(n_agents=n, noise=float(eta), radius=1.0 if kind == "flock" else 0.0)
    rng = np.random.default_rng(30_000 * n + 101 * seed + int(eta * 1000))
    s = model.initial(rng)
    for _ in range(BURN_IN):
        s = model.step(s, rng)
    phi = np.empty(N_MACRO)
    feats = []
    for t in range(N_MACRO):
        for _ in range(TAU):
            s = model.step(s, rng)
        phi[t] = np.hypot(np.cos(s[:, 2]).mean(), np.sin(s[:, 2]).mean())
        feats.append(micro_features(s, model.box))
    return {"n": n, "noise": float(eta), "seed": seed, "kind": kind,
            "phi": phi.tolist(), "feats": np.array(feats).tolist()}


def aggregation_law(phi: np.ndarray, eta: float, n: int) -> np.ndarray:
    """E[phi_{t+TAU} | phi_t] for independent agents.

    Each heading picks up TAU independent uniform kicks, so E[e^{i theta'}] =
    c^TAU e^{i theta}. For large N the mean velocity is then c^TAU V plus
    isotropic Gaussian noise with per-component variance (1 - c^(2 TAU)) / (2N),
    so |V'| is Rice distributed and its mean is the Rice mean.
    """
    from scipy.stats import rice

    c = np.sin(eta / 2) / (eta / 2)
    nu = c**TAU * phi
    sigma = np.sqrt((1 - c ** (2 * TAU)) / (2 * n))
    return rice.mean(nu / sigma, scale=sigma)


def score(run: dict) -> dict:
    from emergence_lab.closure import fit_held_out, predictive_closure
    from emergence_lab.layers import poly

    phi, feats = np.array(run["phi"]), np.array(run["feats"])
    z, zn = phi[:-1, None], phi[1:, None]
    blocks = np.arange(len(z)) // BLOCK
    cut = int(len(z) * 0.7)
    tr, te = np.arange(cut), np.arange(cut, len(z))
    rng = np.random.default_rng(0)
    _, pred = fit_held_out(poly(z, 3), zn, tr, te, rng)
    mse_g = float(((zn[te] - pred) ** 2).mean())
    mse_g0 = float(((zn[te, 0] - aggregation_law(z[te, 0], run["noise"], run["n"])) ** 2).mean())
    pc = predictive_closure(z, zn, feats[:-1], groups=blocks, n_micro_components=feats.shape[1])
    return {"n": run["n"], "noise": run["noise"], "seed": run["seed"], "kind": run["kind"],
            "S": 1 - mse_g / mse_g0, "autonomy": pc["autonomy"], "r2_macro": pc["r2_macro"],
            "r2_full": pc["r2_full"], "pol_mean": float(phi.mean()),
            "chi": float(run["n"] * phi.var())}


def _run_and_score(args):
    return score(_run(args))


def analyze(rows: list[dict], out_dir: Path) -> str:
    def table(kind, key):
        return {e: np.array([r[key] for r in rows if r["kind"] == kind and r["noise"] == e])
                for e in NOISES}

    chi = {e: v.mean() for e, v in table("flock", "chi").items()}
    eta_c = max(chi, key=chi.get)
    lines = [f"Experiment 9: interaction share and closure of polarization (N={N_AGENTS}, "
             f"macro step = {TAU} micro steps)", "",
             f"  susceptibility peak at noise {eta_c}", "",
             f"  {'noise':>6}{'|V| flock':>10}{'S flock':>9}{'S twin':>8}{'auton. flock':>14}"
             f"{'auton. twin':>13}{'R2 φ-only':>11}{'R2 +micro':>11}"]
    S_f, S_t = table("flock", "S"), table("twin", "S")
    A_f, A_t = table("flock", "autonomy"), table("twin", "autonomy")
    R_m, R_f = table("flock", "r2_macro"), table("flock", "r2_full")
    P = table("flock", "pol_mean")
    m = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
    for e in NOISES:
        lines.append(f"  {e:6.2f}{m(P[e]):10.3f}{m(S_f[e]):9.3f}{m(S_t[e]):8.3f}{m(A_f[e]):14.3f}"
                     f"{m(A_t[e]):13.3f}{m(R_m[e]):11.3f}{m(R_f[e]):11.3f}")

    e1 = all(m(S_t[e]) <= 0.1 for e in NOISES) and all(
        m(A_t[e]) >= 0.9 for e in NOISES if np.isfinite(m(A_t[e])))
    e2 = all(m(S_f[e]) >= 0.5 for e in NOISES if e < eta_c)
    a_c, a_0 = m(A_f[eta_c]), m(A_f[NOISES[0]])
    e3 = np.isfinite(a_c) and np.isfinite(a_0) and a_0 - a_c >= 0.1
    lines += ["", "  Pre-registered predictions:",
              f"  {'PASS' if e1 else 'FAIL'}  E1 twin: S <= 0.1 everywhere, autonomy >= 0.9 where defined",
              f"  {'PASS' if e2 else 'FAIL'}  E2 flock: S >= 0.5 below the transition",
              f"  {'PASS' if e3 else 'FAIL'}  E3 flock autonomy drops >= 0.1 at the transition "
              f"[{a_0:.3f} at noise {NOISES[0]} -> {a_c:.3f} at noise {eta_c}]"]
    _plot(rows, eta_c, out_dir / "e9_interaction_closure.png")
    lines.append(f"\n  figure: {out_dir / 'e9_interaction_closure.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(rows, eta_c, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(kind, key):
        vals = [[r[key] for r in rows if r["kind"] == kind and r["noise"] == e] for e in NOISES]
        return np.array([np.nanmean(v) for v in vals]), np.array([np.nanstd(v) for v in vals])

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    pm, _ = series("flock", "pol_mean")
    cm, cs = series("flock", "chi")
    ax[0].plot(NOISES, pm, "o-", label="polarization")
    ax0b = ax[0].twinx()
    ax0b.errorbar(NOISES, cm, yerr=cs, fmt="s--", color="tab:red", capsize=2)
    ax0b.set_ylabel("susceptibility", color="tab:red")
    ax[0].set_title("Flock order and transition")
    for i, (key, title) in enumerate((("S", "Interaction share of the macro law"),
                                      ("autonomy", "Autonomy (closure) of polarization")), start=1):
        for kind, c in (("flock", "tab:blue"), ("twin", "tab:gray")):
            mu, sd = series(kind, key)
            ax[i].errorbar(NOISES, mu, yerr=sd, fmt="o-", color=c, capsize=2,
                           label="flock" if kind == "flock" else "no-interaction twin")
        ax[i].set_title(title)
        ax[i].set_ylim(-0.1, 1.05)
        ax[i].legend(fontsize=8)
    for a in ax:
        a.axvline(eta_c, color="tab:red", ls=":", lw=1)
        a.set_xlabel("noise")
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = 1 if quick else SEEDS
    tasks = [(N_AGENTS, e, s, k) for k in ("flock", "twin") for e in NOISES for s in range(seeds)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"  running {len(tasks)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(_run_and_score, tasks, chunksize=1))
    (out_dir / "e9_interaction_closure_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows, out_dir)


if __name__ == "__main__":
    main()
