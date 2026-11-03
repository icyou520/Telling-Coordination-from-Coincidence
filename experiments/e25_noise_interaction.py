"""Experiment 25: interactions that change the noise, not the mean.

S = 1 - MSE(G)/MSE(G0) compares mean predictions only. If interactions
change how noisy a macro variable is but not its expected next value, S is
blind to them. The proposed fix compares predictive *distributions*:

  D      = E[ log p_G(z' | z) - log p_G0(z' | z) ]   (nats per step, held out)
  S_info = 1 - exp(-2 D)

where p_G is the fitted law of the real system (Gaussian, linear mean,
constant variance) and p_G0 the twin's exact predictive distribution. When
twin and real system have the same noise, S_info = 1 - exp(-x) with
x = (a - b)^2 / (1 - a^2), which shares the sign, the zero and the
weak-coupling limit x of S = x / (1 + x).

System: e23's walkers with springs (gamma = 0.2, theta = 0.1, N = 10) plus
crowding agitation: each walker's noise variance is

  sigma_i^2 = 1 + kappa exp(-r_i / 2),  r_i = sum_{j ~ i} (Delta_j - Delta_i)^2.

Springs and agitation are both interactions; the twin has neither, so its
centre of mass follows N(b z, 1) exactly. In the real system the springs
cancel in the sum (Lean, com_twin_law) but the agitation inflates the noise.

Data-only twin: the mean model of e24 (own + relative states), and squared
residuals regressed on [1, exp(-r_i / 2)]; the twin's variance is the
intercept.

Predictions, fixed before running:

  D1  MSE-based S of the centre of mass within 0.01 of 0 at every kappa.
  D2  S_info of the centre of mass within 0.01 of 0 at kappa = 0, >= 0.05 at
      kappa = 2, and increasing in kappa.
  D3  At kappa = 0, S_info of modes k = 1 and k = 9 within 0.02 of
      1 - exp(-x) (consistency with the closed form).
  D4  Data-only twin: S_info within 0.02 of the exact-twin value for the
      centre of mass at every kappa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.e23_coupled_walkers import laplacian, modes

N = 10
THETA = 0.1
GAMMA = 0.2
KAPPAS = (0.0, 0.25, 0.5, 1.0, 2.0)
T_SIM = 200_000
BURN_IN = 1_000


def crowding(x: np.ndarray) -> np.ndarray:
    """r_i = sum over chain neighbours of squared distance (works on (..., N))."""
    d = np.diff(x, axis=-1) ** 2
    r = np.zeros_like(x)
    r[..., :-1] += d
    r[..., 1:] += d
    return np.exp(-r / 2.0)


def simulate(kappa: float, steps: int, rng) -> np.ndarray:
    m = (1.0 - THETA) * np.eye(N) - GAMMA * laplacian(N)
    noise = rng.standard_normal((BURN_IN + steps, N))
    x = np.zeros(N)
    out = np.empty((steps, N))
    for t in range(BURN_IN + steps):
        x = m @ x + np.sqrt(1.0 + kappa * crowding(x)) * noise[t]
        if t >= BURN_IN:
            out[t - BURN_IN] = x
    return out


def _gauss_ll(z, mean, var):
    return -0.5 * (np.log(2 * np.pi * var) + (z - mean) ** 2 / var)


def shares(y: np.ndarray, b: float, var0: float) -> tuple[float, float]:
    """(S, S_info) of macro series y against twin N(b y, var0), held out."""
    x, z = y[:-1], y[1:]
    h = len(x) // 2
    slope, icpt = np.polyfit(x[:h], z[:h], 1)
    var = np.mean((z[:h] - slope * x[:h] - icpt) ** 2)
    pred = slope * x[h:] + icpt
    s = 1.0 - np.mean((z[h:] - pred) ** 2) / np.mean((z[h:] - b * x[h:]) ** 2)
    d = np.mean(_gauss_ll(z[h:], pred, var) - _gauss_ll(z[h:], b * x[h:], var0))
    return float(s), float(1.0 - np.exp(-2.0 * d))


def fitted_twin(x: np.ndarray) -> tuple[float, float]:
    now, nxt = x[:-1], x[1:]
    cols = np.column_stack([now.ravel(), -(now @ laplacian(N)).ravel()])
    coef = np.linalg.lstsq(cols, nxt.ravel(), rcond=None)[0]
    res2 = (nxt.ravel() - cols @ coef) ** 2
    vcols = np.column_stack([np.ones(res2.size), crowding(now).ravel()])
    vcoef = np.linalg.lstsq(vcols, res2, rcond=None)[0]
    return float(coef[0]), float(vcoef[0])


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    steps = 40_000 if quick else T_SIM
    lam, u = modes(N)
    picks = [0, 1, N - 1]
    rows = {}
    for kappa in KAPPAS:
        print(f"  kappa={kappa} ...", flush=True)
        x = simulate(kappa, steps, np.random.default_rng(25_000 + int(100 * kappa)))
        b_hat, v_hat = fitted_twin(x[: len(x) // 2])
        y = x @ u[:, picks]
        for i, k in enumerate(picks):
            exact = shares(y[:, i], 1.0 - THETA, 1.0)
            est = shares(y[:, i], b_hat, v_hat)
            a = 1.0 - THETA - GAMMA * lam[k]
            xk = (a - (1.0 - THETA)) ** 2 / (1.0 - a * a)
            rows[(kappa, k)] = exact + est[1:] + (1.0 - np.exp(-xk), b_hat, v_hat)
    return analyze(rows, picks, out_dir)


def analyze(rows, picks, out_dir: Path) -> str:
    names = {0: "com", 1: "k=1", N - 1: f"k={N - 1}"}
    lines = [f"Experiment 25: noise-changing interactions (N={N}, theta={THETA}, "
             f"gamma={GAMMA})", "",
             f"  {'kappa':>5}{'mode':>6}{'S':>8}{'S_info':>8}{'S_info fit':>11}"
             f"{'1-exp(-x)':>10}   fitted twin (b, var)"]
    for (kappa, k), (s, si, si_fit, closed, b, v) in rows.items():
        tail = f"   {b:.3f}, {v:.3f}" if k == 0 else ""
        lines.append(f"  {kappa:5.2f}{names[k]:>6}{s:8.4f}{si:8.4f}{si_fit:11.4f}"
                     f"{closed:10.4f}{tail}")
    com = [rows[(kp, 0)] for kp in KAPPAS]
    d1_err = max(abs(r[0]) for r in com)
    d1 = d1_err <= 0.01
    si = [r[1] for r in com]
    d2 = abs(si[0]) <= 0.01 and si[-1] >= 0.05 and all(np.diff(si) > 0)
    d3_err = max(abs(rows[(0.0, k)][1] - rows[(0.0, k)][3]) for k in picks[1:])
    d3 = d3_err <= 0.02
    d4_err = max(abs(r[1] - r[2]) for r in com)
    d4 = d4_err <= 0.02
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(d1)}  D1 MSE-based S(com) within 0.01 of 0 [max |S| {d1_err:.4f}]",
              f"  {tag(d2)}  D2 S_info(com) ~0 at kappa 0, >= 0.05 at kappa 2, increasing "
              f"[{', '.join(f'{v:.3f}' for v in si)}]",
              f"  {tag(d3)}  D3 S_info matches 1 - exp(-x) at kappa 0 [max error {d3_err:.4f}]",
              f"  {tag(d4)}  D4 data-only twin within 0.02 for com [max error {d4_err:.4f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
