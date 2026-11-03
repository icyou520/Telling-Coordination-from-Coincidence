"""Experiment 30: Kuramoto oscillators, identical and heterogeneous.

N = 500 phase oscillators with mean-field coupling and phase noise,

    theta_i' = theta_i + dt (omega_i + K Im(Z e^{-i theta_i})) + sqrt(2 D dt) xi,

Z = mean e^{i theta}, r = |Z|, dt = 0.05, D = 0.5, macro step tau dt = 1.
Identical oscillators (omega = 0) synchronize above K_c = 2D = 1; with
Lorentzian frequencies of width Delta = 0.25 (deterministic quantiles),
above K_c = 2(D + Delta) = 1.5.

The twin (K = 0) rotates freely with Brownian phase noise, so from a micro
state (theta, omega) its exact next mean field is

    Z' = mean e^{i(theta_i + omega_i tau dt + Xi_i)},  Xi_i ~ N(0, 2 D tau dt).

Two versions of the twin's law are compared on the same states:

  macro   G0(r): Rice mean with nu = C r, sigma^2 = (1 - C^2) / (2N),
          C = exp(-(D + Delta) tau dt). Exact for the twin's own states,
          where phases are independent of frequencies.
  micro   G0(x) = E|Z'| from the actual micro state (32 Monte Carlo replicas
          of the twin's exact one-step law).

For identical oscillators the two differ only by the anisotropy of the noise.
For heterogeneous ones they can genuinely differ: in a synchronized state
phases are locked to frequencies, so the twin's macro law is not a function
of r alone. If they disagree, S is ambiguous there.

S uses a cubic G in r fitted on 70 % of the run; both versions are scored
on the same ~300 stored test states.

Predictions, fixed before running:

  K1  Identical: S <= 0.02 at K = 0, and S >= 0.5 for K >= 1.5.
  K2  Identical: |S_macro - S_micro| <= 0.02 at every K.
  K3  Heterogeneous: |S_macro - S_micro| <= 0.05 at every K.
  K4  Heterogeneous: S_micro <= 0.02 at K = 0.

Bug fixed after the first run: micro states were stored at the start of a
macro step instead of the end, so the micro twin was scored against r two
steps ahead. The K = 0 control exposed it (S_micro = +0.07 where the system
is its own twin, so S_micro must be <= 0). K2-K4 failed in that run;
predictions are unchanged for the rerun.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N = 500
DT, D, TAU = 0.05, 0.5, 20
DELTA = 0.25
KS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
MACRO, BURN_IN = 3_000, 500
STORE_EVERY = 3
REPLICAS = 32


def frequencies(hetero: bool) -> np.ndarray:
    if not hetero:
        return np.zeros(N)
    return DELTA * np.tan(np.pi * ((np.arange(N) + 0.5) / N - 0.5))


def simulate(k: float, omega: np.ndarray, rng):
    theta = rng.uniform(-np.pi, np.pi, N)
    r = np.empty(MACRO)
    cut = int(0.7 * (MACRO - 1))
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
            if tt >= cut and tt < MACRO - 1 and (tt - cut) % STORE_EVERY == 0:
                stored[tt] = theta.copy()
    return r, stored


def rice_mean(nu, sigma):
    from scipy.stats import rice

    nu = np.atleast_1d(nu)
    snr = nu / sigma
    out = np.empty_like(nu)
    large = snr > 30.0
    out[~large] = rice.mean(snr[~large], scale=sigma)
    out[large] = nu[large] + sigma ** 2 / (2 * nu[large])
    return out


def g0_micro(theta, omega, rng) -> float:
    xi = rng.standard_normal((REPLICAS, N)) * np.sqrt(2 * D * TAU * DT)
    return float(np.abs(np.exp(1j * (theta + omega * TAU * DT + xi)).mean(1)).mean())


def shares(r, stored, omega, hetero, rng) -> dict:
    x, z = r[:-1], r[1:]
    cut = int(0.7 * len(x))
    coef = np.polyfit(x[:cut], z[:cut], 3)
    idx = np.array(sorted(stored))
    err_g = np.mean((z[idx] - np.polyval(coef, x[idx])) ** 2)
    c = np.exp(-(D + (DELTA if hetero else 0.0)) * TAU * DT)
    g_macro = rice_mean(c * x[idx], np.sqrt((1 - c ** 2) / (2 * N)))
    g_micro = np.array([g0_micro(stored[i], omega, rng) for i in idx])
    return {"S_macro": float(1 - err_g / np.mean((z[idx] - g_macro) ** 2)),
            "S_micro": float(1 - err_g / np.mean((z[idx] - g_micro) ** 2)),
            "r": float(r.mean())}


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    global MACRO
    if quick:
        MACRO = 800
    rows = {}
    for hetero in (False, True):
        omega = frequencies(hetero)
        for k in KS:
            print(f"  {'heterogeneous' if hetero else 'identical'} K={k} ...", flush=True)
            rng = np.random.default_rng(30_000 + int(10 * k) + (1000 if hetero else 0))
            r, stored = simulate(k, omega, rng)
            rows[(hetero, k)] = shares(r, stored, omega, hetero, rng)
    return analyze(rows)


def analyze(rows) -> str:
    lines = [f"Experiment 30: Kuramoto (N={N}, D={D}, macro step 1 time unit)", "",
             f"  {'oscillators':>13}{'K':>5}{'<r>':>7}{'S macro twin':>14}{'S micro twin':>14}"]
    for (hetero, k), v in rows.items():
        lines.append(f"  {'heterogeneous' if hetero else 'identical':>13}{k:5.1f}{v['r']:7.3f}"
                     f"{v['S_macro']:14.4f}{v['S_micro']:14.4f}")
    ident = {k: rows[(False, k)] for k in KS}
    het = {k: rows[(True, k)] for k in KS}
    k1 = ident[0.0]["S_macro"] <= 0.02 and all(ident[k]["S_macro"] >= 0.5 for k in KS if k >= 1.5)
    k2_err = max(abs(v["S_macro"] - v["S_micro"]) for v in ident.values())
    k3_err = max(abs(v["S_macro"] - v["S_micro"]) for v in het.values())
    k4 = het[0.0]["S_micro"] <= 0.02
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(k1)}  K1 identical: S <= 0.02 at K=0 [{ident[0.0]['S_macro']:.4f}], "
              f">= 0.5 for K >= 1.5",
              f"  {tag(k2_err <= 0.02)}  K2 identical: macro vs micro twin within 0.02 "
              f"[max {k2_err:.4f}]",
              f"  {tag(k3_err <= 0.05)}  K3 heterogeneous: macro vs micro twin within 0.05 "
              f"[max {k3_err:.4f}]",
              f"  {tag(k4)}  K4 heterogeneous: S_micro <= 0.02 at K=0 [{het[0.0]['S_micro']:.4f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
