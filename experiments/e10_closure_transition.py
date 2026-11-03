"""Experiment 10: does the flock's macro layer lose closure *at* the transition?

Experiment 9 found that polarization is a closed, interaction-made macro
variable deep in the ordered phase, but leaks (micro detail helps predict its
future) near the susceptibility peak. With one size and coarse noise steps the
dip was broad, so it could be a generic feature of the ordered phase rather
than of the transition, or simply a sign that the macro law needs more
nonlinearity.

Tests:
  * two sizes, N = 400 and 1600 (density fixed): the finite-size transition
    point moves with N, so a dip tied to the transition should move with it;
  * a finer noise grid and 6 seeds;
  * the macro law is fitted with polynomials in phi of degree 2 and of degree 4,
    so a dip caused only by an under-flexible macro law disappears at degree 4.

Autonomy depth at noise e = mean autonomy at the two lowest noises - autonomy(e).

Predictions, fixed before running:

  G1  For each N, the noise of minimum mean autonomy (degree 2) is within 0.5
      of the noise of maximum mean susceptibility.
  G2  If the susceptibility peaks of the two sizes differ by >= 0.25, the
      autonomy minimum shifts in the same direction (otherwise: untestable).
  G3  Depth at the susceptibility peak >= 0.1 at both N with the degree-2 law.
  G4  Depth at the susceptibility peak >= 0.1 at both N with the degree-4 law.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e9_interaction_closure import BLOCK, _run

SIZES = (400, 1600)
NOISES = np.round(np.arange(2.0, 5.01, 0.25), 3)
SEEDS = 6
DEGREES = (2, 4)


def score(run: dict) -> dict:
    from emergence_lab.closure import predictive_closure

    phi, feats = np.array(run["phi"]), np.array(run["feats"])
    dphi = phi[1:] - phi[:-1]
    blocks = np.arange(len(dphi)) // BLOCK
    out = {"n": run["n"], "noise": run["noise"], "seed": run["seed"],
           "pol_mean": float(phi.mean()), "chi": float(run["n"] * phi.var())}
    for d in DEGREES:
        z = np.column_stack([phi[:-1] ** k for k in range(1, d + 1)])
        pc = predictive_closure(z, dphi, feats[:-1], groups=blocks, degree=1,
                                n_micro_components=feats.shape[1], predict_change=False)
        out[f"autonomy{d}"] = pc["autonomy"]
        out[f"r2_macro{d}"] = pc["r2_macro"]
        out[f"r2_full{d}"] = pc["r2_full"]
    return out


def _run_and_score(args):
    return score(_run(args))


def _mean(a) -> float:
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def analyze(rows: list[dict], out_dir: Path) -> str:
    def curve(n, key):
        return np.array([_mean([r[key] for r in rows if r["n"] == n and r["noise"] == e])
                         for e in NOISES])

    lines = ["Experiment 10: closure of polarization across the transition, two sizes", ""]
    eta_c, eta_min, depth = {}, {}, {}
    for n in SIZES:
        chi = curve(n, "chi")
        ic = int(np.nanargmax(chi))
        eta_c[n] = float(NOISES[ic])
        a = {d: curve(n, f"autonomy{d}") for d in DEGREES}
        filled = np.where(np.isfinite(a[2]), a[2], np.inf)
        eta_min[n] = float(NOISES[int(np.argmin(filled))])
        depth[n] = {d: _mean(a[d][:2]) - a[d][ic] for d in DEGREES}
        r2m, r2f = curve(n, "r2_macro2"), curve(n, "r2_full2")
        lines += [f"  N = {n}: susceptibility peak at noise {eta_c[n]}, "
                  f"autonomy minimum at noise {eta_min[n]}",
                  f"  {'noise':>6}{'|V|':>7}{'chi':>8}{'auton. d2':>11}{'auton. d4':>11}"
                  f"{'R2 φ d2':>9}{'R2 +micro':>11}"]
        pol = curve(n, "pol_mean")
        for i, e in enumerate(NOISES):
            mark = " <- chi peak" if i == ic else ""
            lines.append(f"  {e:6.2f}{pol[i]:7.3f}{chi[i]:8.2f}{a[2][i]:11.3f}{a[4][i]:11.3f}"
                         f"{r2m[i]:9.3f}{r2f[i]:11.3f}{mark}")
        lines.append("")

    worst = max(abs(r[f"r2_{k}{d}"]) for r in rows for k in ("macro", "full") for d in DEGREES)
    g1 = all(abs(eta_min[n] - eta_c[n]) <= 0.5 for n in SIZES)
    shift_c = eta_c[SIZES[1]] - eta_c[SIZES[0]]
    shift_m = eta_min[SIZES[1]] - eta_min[SIZES[0]]
    g2 = None if abs(shift_c) < 0.25 else bool(np.sign(shift_m) == np.sign(shift_c))
    g3 = all(np.isfinite(depth[n][2]) and depth[n][2] >= 0.1 for n in SIZES)
    g4 = all(np.isfinite(depth[n][4]) and depth[n][4] >= 0.1 for n in SIZES)
    tag = lambda ok: "N/A " if ok is None else ("PASS" if ok else "FAIL")
    lines += [f"  sanity: largest |R2| over all fits = {worst:.3f}", "",
              "  Pre-registered predictions:",
              f"  {tag(g1)}  G1 autonomy minimum within 0.5 of the chi peak "
              + ", ".join(f"[N={n}: {eta_min[n]} vs {eta_c[n]}]" for n in SIZES),
              f"  {tag(g2)}  G2 minimum shifts with the peak "
              f"[peak shift {shift_c:+.2f}, minimum shift {shift_m:+.2f}]",
              f"  {tag(g3)}  G3 depth >= 0.1 at the peak, degree-2 law "
              + ", ".join(f"[N={n}: {depth[n][2]:.3f}]" for n in SIZES),
              f"  {tag(g4)}  G4 depth >= 0.1 at the peak, degree-4 law "
              + ", ".join(f"[N={n}: {depth[n][4]:.3f}]" for n in SIZES)]
    _plot(rows, eta_c, out_dir / "e10_closure_transition.png")
    lines.append(f"\n  figure: {out_dir / 'e10_closure_transition.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(rows, eta_c, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {SIZES[0]: "tab:blue", SIZES[1]: "tab:orange"}
    for n in SIZES:
        def stats(key):
            vals = [[r[key] for r in rows if r["n"] == n and r["noise"] == e] for e in NOISES]
            return (np.array([_mean(v) for v in vals]),
                    np.array([np.nanstd(v) if np.isfinite(v).any() else np.nan for v in vals]))
        mu, sd = stats("chi")
        ax[0].errorbar(NOISES, mu / mu.max(), yerr=sd / mu.max(), fmt="o-", capsize=2,
                       color=colors[n], label=f"N={n}")
        for i, d in enumerate(DEGREES, start=1):
            mu, sd = stats(f"autonomy{d}")
            ax[i].errorbar(NOISES, mu, yerr=sd, fmt="o-", capsize=2, color=colors[n], label=f"N={n}")
        for a in ax:
            a.axvline(eta_c[n], color=colors[n], ls=":", lw=1)
    ax[0].set_title("Susceptibility (normalized)")
    ax[1].set_title("Autonomy of polarization, degree-2 macro law")
    ax[2].set_title("Autonomy of polarization, degree-4 macro law")
    for a in ax:
        a.set_xlabel("noise")
        a.grid(alpha=0.2)
        a.legend(fontsize=8)
    for a in ax[1:]:
        a.set_ylim(0.3, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = 1 if quick else SEEDS
    tasks = [(n, e, s, "flock") for n in sorted(SIZES, reverse=True) for e in NOISES
             for s in range(seeds)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"  running {len(tasks)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(_run_and_score, tasks, chunksize=1))
    (out_dir / "e10_closure_transition_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows, out_dir)


if __name__ == "__main__":
    main()
