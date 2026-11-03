"""Experiment 29: the 2D Ising model across its critical point.

Unlike Curie-Weiss, the magnetization of the 2D Ising model is not closed
(its future depends on domain structure) and the model has a true critical
point (T_c = 2.269). L = 32 periodic lattice, J = 1, heat-bath dynamics on a
checkerboard: each macro step updates the black then the white sublattice,
and each site in the active sublattice is redrawn with probability q = 0.2
(+1 with probability 1 / (1 + exp(-2 beta h)), h = sum of 4 neighbours).
Same-colour sites are not neighbours, so each half-step is exact.

Twin (beta = 0): every site is redrawn with probability q as a fair coin, so
its exact macro law is E[m' | m] = (1 - q) m.

Data-only twin: logistic regression of redrawn spins on their neighbour sum,
P(+1) = 1 / (1 + exp(-(a + b h))); the twin redraws with probability
p0 = 1 / (1 + exp(-a)), so G0 = (1 - q) m + q (2 p0 - 1).

S uses a cubic G in m fitted on 70 % of the run and scored on the rest.

Predictions, fixed before running:

  I1  |S| <= 0.01 at beta = 0.
  I2  S strictly increasing as T decreases from infinity to 2.3.
  I3  S >= 0.5 at T = 2.1 and 1.8 (ordered phase).
  I4  Data-only twin within 0.02 of the exact-twin S at every temperature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

L = 32
Q = 0.2
TEMPS = (np.inf, 5.0, 3.5, 2.8, 2.5, 2.3, 2.1, 1.8)
STEPS = 20_000
BURN_IN = 2_000
SAMPLE_EVERY = 20


def _nbr(s):
    return np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)


def simulate(beta: float, rng):
    s = np.ones((L, L), dtype=np.int8)
    colour = (np.add.outer(np.arange(L), np.arange(L)) % 2).astype(bool)
    m = np.empty(STEPS)
    hs, news = [], []
    for t in range(BURN_IN + STEPS):
        for c in (colour, ~colour):
            h = _nbr(s.astype(np.int16))
            new = np.where(rng.random((L, L)) < 1.0 / (1.0 + np.exp(-2.0 * beta * h)), 1, -1)
            mask = c & (rng.random((L, L)) < Q)
            if t >= BURN_IN and t % SAMPLE_EVERY == 0:
                hs.append(h[mask])
                news.append(new[mask])
            s = np.where(mask, new, s).astype(np.int8)
        if t >= BURN_IN:
            m[t - BURN_IN] = s.mean()
    return m, np.concatenate(hs), np.concatenate(news)


def fit_p0(h: np.ndarray, new: np.ndarray) -> float:
    """Intercept-only twin from a logistic fit of new spins on neighbour sum (Newton)."""
    x = np.column_stack([np.ones(h.size), h.astype(float)])
    y = (new > 0).astype(float)
    w = np.zeros(2)
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-x @ w))
        grad = x.T @ (y - p)
        hess = (x * (p * (1 - p))[:, None]).T @ x + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(1.0 / (1.0 + np.exp(-w[0])))


def share(m: np.ndarray, p0: float) -> float:
    x, z = m[:-1], m[1:]
    cut = int(0.7 * len(x))
    coef = np.polyfit(x[:cut], z[:cut], 3)
    mse_g = np.mean((z[cut:] - np.polyval(coef, x[cut:])) ** 2)
    g0 = (1 - Q) * x[cut:] + Q * (2 * p0 - 1)
    return float(1.0 - mse_g / np.mean((z[cut:] - g0) ** 2))


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    global STEPS
    if quick:
        STEPS = 4_000
    rows = {}
    for temp in TEMPS:
        beta = 0.0 if np.isinf(temp) else 1.0 / temp
        print(f"  T={temp} ...", flush=True)
        m, h, new = simulate(beta, np.random.default_rng(29_000 + int(100 * beta)))
        p0 = fit_p0(h, new)
        rows[temp] = {"abs_m": float(np.abs(m).mean()), "S": share(m, 0.5),
                      "S_fit": share(m, p0), "p0": p0}
    return analyze(rows)


def analyze(rows) -> str:
    lines = [f"Experiment 29: 2D Ising (L={L}, q={Q}, T_c = 2.269)", "",
             f"  {'T':>6}{'<|m|>':>8}{'S exact twin':>14}{'S data twin':>13}{'p0 fitted':>11}"]
    for t, r in rows.items():
        lines.append(f"  {t:6.2f}{r['abs_m']:8.3f}{r['S']:14.4f}{r['S_fit']:13.4f}{r['p0']:11.4f}")
    s = [rows[t]["S"] for t in TEMPS]
    i1 = abs(s[0]) <= 0.01
    above = [rows[t]["S"] for t in TEMPS if t >= 2.3]
    i2 = all(np.diff(above) > 0)
    i3 = rows[2.1]["S"] >= 0.5 and rows[1.8]["S"] >= 0.5
    i4_err = max(abs(r["S"] - r["S_fit"]) for r in rows.values())
    i4 = i4_err <= 0.02
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(i1)}  I1 |S| <= 0.01 at beta 0 [{s[0]:.4f}]",
              f"  {tag(i2)}  I2 S increasing as T falls to 2.3 "
              f"[{', '.join(f'{v:.3f}' for v in above)}]",
              f"  {tag(i3)}  I3 S >= 0.5 in the ordered phase "
              f"[T=2.1: {rows[2.1]['S']:.3f}, T=1.8: {rows[1.8]['S']:.3f}]",
              f"  {tag(i4)}  I4 data-only twin within 0.02 [max error {i4_err:.4f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
