"""Experiment 28: exact interaction share and Psi in a multistable system.

Curie-Weiss Ising model (all-to-all coupling J = 1, N = 50 spins), heat-bath
dynamics with random sequential updates: a random spin is redrawn as +1 with
probability 1 / (1 + exp(-2 beta h)), h = (sum of the other spins) / N. The
number of up spins k is an exact Markov chain (lumpable), so the macro law
of m = (2k - N) / N, its stationary distribution and everything below are
computed exactly from the (N+1)-state transition matrix. One macro step is
one sweep (tau = N updates). The twin is the same model at beta = 0 (every
redrawn spin is a fair coin).

  S      = 1 - E_pi[Var(m' | m)] / E_pi[(m' - G0(m))^2],  G0 the twin's law
  Psi    = I(m; m') - sum_i I(s_i; m')

Psi is exact too: the stationary measure is exchangeable and the dynamics
are permutation-symmetric, so m' is independent of s_i given m, and
P(s_i = +1, m') = sum_k pi(k) (k / N) P^tau(k, m').

Below beta = 1 the model has one stable state (m = 0); above it, two
(m = +-m0) with rare switches.

Predictions, fixed before running:

  C1  Exact S = 0 at beta = 0 and strictly increasing in beta over the grid.
  C2  Spin-level simulation (20 000 sweeps; G as the per-level mean of m' on
      the first half; scored on the second half) within 0.02 of exact S at
      every beta.
Exploratory: exact Psi across the transition.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

N = 50
BETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
SWEEPS = 20_000
BURN_IN = 1_000


def p_up(beta: float, others: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-2.0 * beta * others / N))


def transition(beta: float) -> np.ndarray:
    k = np.arange(N + 1)
    mag = 2 * k - N
    p = np.zeros((N + 1, N + 1))
    down = (k / N) * (1.0 - p_up(beta, mag - 1.0))
    up = ((N - k) / N) * p_up(beta, mag + 1.0)
    p[k[1:], k[1:] - 1] = down[1:]
    p[k[:-1], k[:-1] + 1] = up[:-1]
    p[k, k] = 1.0 - p.sum(1)
    return np.linalg.matrix_power(p, N)


def stationary(p: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(p.T)
    pi = np.abs(np.real(v[:, np.argmin(np.abs(w - 1.0))]))
    return pi / pi.sum()


def _mi(joint: np.ndarray) -> float:
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def exact(beta: float) -> dict:
    m = (2 * np.arange(N + 1) - N) / N
    pt, p0 = transition(beta), transition(0.0)
    pi = stationary(pt)
    g = pt @ m
    g0 = p0 @ m
    mse_g = float(pi @ (pt @ m ** 2 - g ** 2))
    mse_g0 = float(pi @ (pt @ m ** 2 - 2 * g * g0 + g0 ** 2))
    joint = pi[:, None] * pt
    frac = np.arange(N + 1) / N
    spin = np.vstack([(pi * frac) @ pt, (pi * (1 - frac)) @ pt])
    psi = _mi(joint) - N * _mi(spin)
    return {"S": 1.0 - mse_g / mse_g0, "psi": psi, "g0": g0, "abs_m": float(pi @ np.abs(m))}


def simulate(beta: float, rng) -> np.ndarray:
    s = [int(v) for v in rng.choice([-1, 1], N)]
    total = sum(s)
    out = np.empty(SWEEPS, dtype=np.int64)
    prob = {o: 1.0 / (1.0 + math.exp(-2.0 * beta * o / N)) for o in range(-N, N + 1)}
    for sweep in range(BURN_IN + SWEEPS):
        idx = rng.integers(0, N, N).tolist()
        u = rng.random(N).tolist()
        for i, r in zip(idx, u):
            new = 1 if r < prob[total - s[i]] else -1
            total += new - s[i]
            s[i] = new
        if sweep >= BURN_IN:
            out[sweep - BURN_IN] = (total + N) // 2
    return out


def share_sim(k: np.ndarray, g0: np.ndarray) -> float:
    m = (2 * k - N) / N
    x, z = k[:-1], m[1:]
    h = len(x) // 2
    sums = np.bincount(x[:h], z[:h], N + 1)
    counts = np.bincount(x[:h], minlength=N + 1)
    seen = np.flatnonzero(counts)
    table = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    nearest = seen[np.abs(seen[None, :] - np.arange(N + 1)[:, None]).argmin(1)]
    table = table[nearest]
    xt, zt = x[h:], z[h:]
    return float(1.0 - np.mean((zt - table[xt]) ** 2) / np.mean((zt - g0[xt]) ** 2))


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    global SWEEPS
    if quick:
        SWEEPS = 4_000
    rows = {}
    for beta in BETAS:
        print(f"  beta={beta} ...", flush=True)
        ex = exact(beta)
        k = simulate(beta, np.random.default_rng(28_000 + int(100 * beta)))
        rows[beta] = ex | {"S_sim": share_sim(k, ex["g0"]),
                           "switches": int(np.sum(np.diff(np.sign(2 * k - N)) != 0))}
    return analyze(rows)


def analyze(rows) -> str:
    lines = [f"Experiment 28: Curie-Weiss, exact S and Psi (N={N}, one sweep per macro step)", "",
             f"  {'beta':>5}{'<|m|>':>8}{'S exact':>9}{'S sim':>8}{'Psi':>9}{'sign changes':>14}"]
    for b, r in rows.items():
        lines.append(f"  {b:5.2f}{r['abs_m']:8.3f}{r['S']:9.4f}{r['S_sim']:8.4f}{r['psi']:9.4f}"
                     f"{r['switches']:14d}")
    s = [rows[b]["S"] for b in BETAS]
    c1 = abs(s[0]) < 1e-9 and all(np.diff(s) > 0)
    c2_err = max(abs(r["S"] - r["S_sim"]) for r in rows.values())
    c2 = c2_err <= 0.02
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(c1)}  C1 exact S zero at beta 0 and strictly increasing",
              f"  {tag(c2)}  C2 simulation within 0.02 of exact [max error {c2_err:.4f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
