"""Experiment 27: the twin as a part in isolation, without a modelling convention.

e24 identified the twin by writing interactions as functions of *relative*
states, kappa (x_j - x_i). That is a convention: the same spring force can be
written as gamma x_j (absolute neighbour state) plus -gamma x_i booked as
"own dynamics", and then the twin changes. The convention-free definition of
the twin is a part with no neighbours at all. If parts differ in how many
neighbours they have, data identify it: fit

    x_i' = (alpha0 + alpha1 deg_i) x_i + beta sum_{j ~ i} x_j + noise

and extrapolate the own-state coefficient to deg = 0 (twin slope alpha0).

Systems: walkers x' = M x + eta with M = (1 - theta) I - gamma L_graph, on
the open chain (N = 10, degrees 1 and 2) and on a connected random graph
(N = 20, edge probability 0.2, degrees vary widely). On any graph the
centre of mass is a zero mode of L, so its exact S is 0; internal modes have
the closed form of e23 with the graph's Laplacian eigenvalues.

Predictions, fixed before running (theta = 0.1, T = 100 000, tau = 1):

  I1  Absolute-state convention with a constant own coefficient (no degree
      term) gives a false positive: S(com) >= 0.05 at gamma = 0.2 on both
      graphs.
  I2  Degree-extrapolated twin: S within 0.02 of exact for the centre of
      mass and for the slowest and fastest internal modes, at every gamma, on
      both graphs.
  I3  Relative-state twin (e24) on the random graph: within 0.02 as well.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.e23_coupled_walkers import laplacian
from experiments.e24_estimated_twin import share_heldout

THETA = 0.1
GAMMAS = (0.0, 0.05, 0.1, 0.2)
T_SIM = 100_000
BURN_IN = 1_000


def random_graph(n: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        a = np.triu(rng.random((n, n)) < p, 1)
        a = (a | a.T).astype(float)
        lap = np.diag(a.sum(1)) - a
        if np.sum(np.linalg.eigvalsh(lap) < 1e-9) == 1:
            return lap


def simulate(lap: np.ndarray, gamma: float, steps: int, rng) -> np.ndarray:
    n = len(lap)
    m = (1.0 - THETA) * np.eye(n) - gamma * lap
    noise = rng.standard_normal((BURN_IN + steps, n))
    x = np.zeros(n)
    out = np.empty((steps, n))
    for t in range(BURN_IN + steps):
        x = m @ x + noise[t]
        if t >= BURN_IN:
            out[t - BURN_IN] = x
    return out


def twins(x: np.ndarray, lap: np.ndarray) -> dict[str, float]:
    adj = np.diag(np.diag(lap)) - lap
    deg = np.diag(lap)
    now, nxt = x[:-1], x[1:]
    y = nxt.ravel()
    own = now.ravel()
    own_deg = (now * deg).ravel()
    nbr_abs = (now @ adj).ravel()
    nbr_rel = -(now @ lap).ravel()
    fit = lambda cols: np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0][0]
    return {"absolute": fit([own, nbr_abs]), "isolation": fit([own, own_deg, nbr_abs]),
            "relative": fit([own, nbr_rel])}


def share_exact(lam: float, gamma: float) -> float:
    a, b = 1.0 - THETA - gamma * lam, 1.0 - THETA
    d = (a - b) ** 2
    return 0.0 if d == 0.0 else d / ((1.0 - a * a) + d)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    steps = 20_000 if quick else T_SIM
    graphs = {"chain": laplacian(10), "random": random_graph(20, 0.2, seed=27)}
    rows = {}
    for gname, lap in graphs.items():
        lam, u = np.linalg.eigh(lap)
        picks = [0, 1, len(lap) - 1]
        print(f"  {gname} (degrees {sorted(set(np.diag(lap).astype(int)))}) ...", flush=True)
        for g in GAMMAS:
            x = simulate(lap, g, steps, np.random.default_rng(27_000 + int(1000 * g) + len(lap)))
            slopes = twins(x[: len(x) // 2], lap)
            y = x @ u[:, picks]
            for i, k in enumerate(picks):
                rows[(gname, g, i)] = {"exact": share_exact(lam[k], g),
                                       **{key: share_heldout(y[:, i], b)
                                          for key, b in slopes.items()},
                                       "slopes": slopes}
    return analyze(rows)


def analyze(rows) -> str:
    names = ["com", "slowest", "fastest"]
    lines = [f"Experiment 27: the twin as a part in isolation (theta={THETA}, tau=1)", "",
             f"  {'graph':>7}{'gamma':>6}{'mode':>9}{'exact':>8}{'absolute':>10}{'isolation':>11}"
             f"{'relative':>10}   twin slopes (abs / iso / rel)"]
    for (gname, g, i), r in rows.items():
        s = r["slopes"]
        tail = (f"   {s['absolute']:.3f} / {s['isolation']:.3f} / {s['relative']:.3f}"
                if i == 0 else "")
        lines.append(f"  {gname:>7}{g:6.2f}{names[i]:>9}{r['exact']:8.4f}{r['absolute']:10.4f}"
                     f"{r['isolation']:11.4f}{r['relative']:10.4f}{tail}")
    i1_vals = [rows[(gn, 0.2, 0)]["absolute"] for gn in ("chain", "random")]
    i1 = min(i1_vals) >= 0.05
    i2_err = max(abs(r["isolation"] - r["exact"]) for r in rows.values())
    i2 = i2_err <= 0.02
    i3_err = max(abs(r["relative"] - r["exact"]) for (gn, _, _), r in rows.items() if gn == "random")
    i3 = i3_err <= 0.02
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(i1)}  I1 absolute convention false positive S(com) >= 0.05 at gamma 0.2 "
              f"[chain {i1_vals[0]:.3f}, random {i1_vals[1]:.3f}]",
              f"  {tag(i2)}  I2 isolation twin within 0.02 everywhere [max error {i2_err:.4f}]",
              f"  {tag(i3)}  I3 relative twin within 0.02 on random graph [max error {i3_err:.4f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
