"""Experiment 34: the projected interaction share on the case where S failed.

e30 found S ambiguous for Kuramoto oscillators with heterogeneous
frequencies: the twin is not closed on r, and its macro (Rice) law and its
micro-informed prediction gave S values up to 0.10 apart (K3 FAIL); even at
K = 0 the micro-informed S was -0.06. e33 introduced the projected share

    G0_proj(r) = cubic fit of the micro-informed twin prediction g0(x_t)
                 against r_t over the training states,
    S_proj     = 1 - MSE(G) / MSE(G0_proj),

which worked on fish. Where the twin's exact law is already a polynomial in
the macro variable of degree <= that of the fit (coupled walkers,
Curie-Weiss, Ising), G0_proj equals it and S_proj = S identically, so those
results carry over unchanged. Only the walkers are rechecked numerically.

Kuramoto set-up exactly as e30 (N = 500, D = 0.5, dt = 0.05, tau = 1 time
unit, Lorentzian width 0.25), except that micro states are stored every 3
macro steps over the whole run (training and test) and g0 uses 64 replicas.

Predictions, fixed before running:

  P1  K = 0: |S_proj| <= 0.02 for identical and heterogeneous oscillators.
  P2  Identical: |S_proj - S_macro| <= 0.02 at every K (the twin is closed
      there, so S_proj must agree with the exact twin law).
  P3  Heterogeneous: fitting G0_proj on test instead of training states
      changes S_proj by <= 0.05 at every K.
  P4  Heterogeneous: S_proj >= S_micro - 0.01 at every K.
  P5  Heterogeneous: S_proj >= 0.9 for K >= 3 (well above K_c = 1.5).
  P6  Walkers (e24 set-up, theta = 0.1 and 0, fitted twin, long runs):
      S_proj within 0.02 of the closed form for every mode and coupling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.e24_estimated_twin import (GAMMAS, N as N_WALK, share_exact, simulate as
                                            simulate_walkers, twin_slopes)
from experiments.e23_coupled_walkers import modes
from experiments.e30_kuramoto import (BURN_IN, D, DELTA, DT, KS, MACRO, N, TAU, frequencies,
                                      rice_mean)

STORE_EVERY = 3
REPLICAS = 64
T_WALK = 200_000


def simulate(k, omega, rng):
    theta = rng.uniform(-np.pi, np.pi, N)
    r = np.empty(MACRO)
    stored = {}
    amp = np.sqrt(2 * D * DT)
    for t in range(BURN_IN + MACRO):
        tt = t - BURN_IN
        for _ in range(TAU):
            z = np.exp(1j * theta).mean()
            theta = theta + DT * (omega + k * np.imag(z * np.exp(-1j * theta))) \
                + amp * rng.standard_normal(N)
        if tt >= 0:
            r[tt] = np.abs(np.exp(1j * theta).mean())
            if tt < MACRO - 1 and tt % STORE_EVERY == 0:
                stored[tt] = theta.copy()
    return r, stored


def g0_micro(thetas, omega, rng):
    xi = rng.standard_normal((len(thetas), REPLICAS, N)) * np.sqrt(2 * D * TAU * DT)
    ph = thetas[:, None, :] + omega * TAU * DT + xi
    return np.abs(np.exp(1j * ph).mean(-1)).mean(-1)


def shares(r, stored, omega, hetero, rng):
    x, z = r[:-1], r[1:]
    cut = int(0.7 * len(x))
    idx = np.array(sorted(stored))
    tr, te = idx[idx < cut], idx[idx >= cut]
    g0 = {}
    for part in (tr, te):
        for a in range(0, len(part), 200):
            chunk = part[a:a + 200]
            for i, v in zip(chunk, g0_micro(np.array([stored[i] for i in chunk]), omega, rng)):
                g0[i] = v
    g0_tr, g0_te = np.array([g0[i] for i in tr]), np.array([g0[i] for i in te])
    mse = lambda pred: np.mean((z[te] - pred) ** 2)
    err_g = mse(np.polyval(np.polyfit(x[:cut], z[:cut], 3), x[te]))
    c = np.exp(-(D + (DELTA if hetero else 0.0)) * TAU * DT)
    g_macro = rice_mean(c * x[te], np.sqrt((1 - c ** 2) / (2 * N)))
    return {"S_proj": float(1 - err_g / mse(np.polyval(np.polyfit(x[tr], g0_tr, 3), x[te]))),
            "S_proj_alt": float(1 - err_g / mse(np.polyval(np.polyfit(x[te], g0_te, 3), x[te]))),
            "S_micro": float(1 - err_g / mse(g0_te)),
            "S_macro": float(1 - err_g / mse(g_macro)), "r": float(r.mean())}


def walkers(theta, gamma, rng):
    """Projected S per mode with the fitted (relative-state) twin; linear laws as in e24."""
    lam, u = modes(N_WALK)
    picks = [0, 1, N_WALK - 1]
    x = simulate_walkers(gamma, theta, T_WALK, rng)
    b = twin_slopes(x[: len(x) // 2])["fitted"]
    out = []
    for j in picks:
        y = x @ u[:, j]
        g0 = (x[:-1] * b) @ u[:, j]
        xs, z = y[:-1], y[1:]
        h = len(xs) // 2
        g = np.polyval(np.polyfit(xs[:h], z[:h], 1), xs[h:])
        proj = np.polyval(np.polyfit(xs[:h], g0[:h], 1), xs[h:])
        s = 1 - np.mean((z[h:] - g) ** 2) / np.mean((z[h:] - proj) ** 2)
        out.append((float(s), share_exact(lam[j], gamma, theta)))
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    rows = {}
    for hetero in (False, True):
        omega = frequencies(hetero)
        for k in KS:
            print(f"  {'heterogeneous' if hetero else 'identical'} K={k} ...", flush=True)
            rng = np.random.default_rng(34_000 + int(10 * k) + (1000 if hetero else 0))
            r, stored = simulate(k, omega, rng)
            rows[(hetero, k)] = shares(r, stored, omega, hetero, rng)
    walk = {(th, g): walkers(th, g, np.random.default_rng(34_500 + int(1000 * g) + int(100 * th)))
            for th in (0.1, 0.0) for g in GAMMAS}
    return analyze(rows, walk)


def analyze(rows, walk) -> str:
    lines = [f"Experiment 34: projected interaction share, Kuramoto (N={N}) and walkers", "",
             f"  {'oscillators':>13}{'K':>5}{'<r>':>7}{'S_proj':>9}{'alt states':>12}"
             f"{'S_micro':>9}{'S_macro':>9}"]
    for (hetero, k), v in rows.items():
        lines.append(f"  {'heterogeneous' if hetero else 'identical':>13}{k:5.1f}{v['r']:7.3f}"
                     f"{v['S_proj']:9.4f}{v['S_proj_alt']:12.4f}{v['S_micro']:9.4f}{v['S_macro']:9.4f}")
    ident = {k: rows[(False, k)] for k in KS}
    het = {k: rows[(True, k)] for k in KS}
    p1 = abs(ident[0.0]["S_proj"]) <= 0.02 and abs(het[0.0]["S_proj"]) <= 0.02
    p2 = max(abs(v["S_proj"] - v["S_macro"]) for v in ident.values())
    p3 = max(abs(v["S_proj"] - v["S_proj_alt"]) for v in het.values())
    p4 = min(v["S_proj"] - v["S_micro"] for v in het.values())
    p5 = min(het[k]["S_proj"] for k in KS if k >= 3)
    p6 = max(abs(s - e) for v in walk.values() for s, e in v)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", f"  Walkers: max |S_proj - closed form| over theta, gamma, modes = {p6:.4f}",
              "", "  Pre-registered predictions:",
              f"  {tag(p1)}  P1 K=0: |S_proj| <= 0.02 [identical {ident[0.0]['S_proj']:.4f}, "
              f"heterogeneous {het[0.0]['S_proj']:.4f}]",
              f"  {tag(p2 <= 0.02)}  P2 identical: S_proj within 0.02 of exact twin law [max {p2:.4f}]",
              f"  {tag(p3 <= 0.05)}  P3 heterogeneous: state dependence <= 0.05 [max {p3:.4f}]",
              f"  {tag(p4 >= -0.01)}  P4 heterogeneous: S_proj >= S_micro - 0.01 [min diff {p4:.4f}]",
              f"  {tag(p5 >= 0.9)}  P5 heterogeneous: S_proj >= 0.9 for K >= 3 [min {p5:.4f}]",
              f"  {tag(p6 <= 0.02)}  P6 walkers within 0.02 of closed form [max {p6:.4f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
