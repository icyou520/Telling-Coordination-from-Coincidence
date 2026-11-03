"""Experiment 23: interaction share on coupled random walkers (closed form).

Liardi et al. (2025, arXiv:2504.10372) showed that Psi calls the centre of
mass of *non-interacting* random walkers causally emergent and decoupled, and
left open how to separate interaction-made emergence from mere statistical
autonomy. Here we compute our interaction share S exactly on their model.

Model (stationary version, theta > 0), N walkers on an open chain:

    Delta(t+1) = M Delta(t) + eta(t),   M = (1 - theta) I - gamma L,

with L the path-graph Laplacian and eta ~ N(0, I). Expanding L into its
orthonormal eigenmodes u_k (eigenvalue lambda_k, lambda_0 = 0 for the centre
of mass), each mode y_k = u_k . Delta is an exact AR(1) process

    y_k(t+1) = a_k y_k(t) + noise,      a_k = 1 - theta - gamma lambda_k.

The no-interaction twin (gamma = 0) gives every mode the aggregation law
y(t+tau) = b^tau y(t) with b = 1 - theta. With S = 1 - MSE(G)/MSE(G0) on the
stationary process at horizon tau:

    S_k(tau) = (a_k^tau - b^tau)^2 / [(1 - a_k^(2 tau)) + (a_k^tau - b^tau)^2].

So S = 0 for the centre of mass at every gamma, N, theta, tau (internal forces
cancel in the sum), and S > 0 for internal modes once gamma > 0, growing as
gamma^2 for weak coupling.

Predictions, fixed before running:

  W1  Simulated S of the centre of mass within 0.01 of 0 at every gamma.
  W2  Exact Psi of the centre of mass > 0 at gamma = 0 for N = 10 and 50
      (reproduces the false positive in the stationary model).
  W3  Exact S of the shortest-wavelength mode (k = N-1) is 0 at gamma = 0 and
      strictly increasing in gamma.
  W4  Simulated S within 0.02 of the closed form for every mode and gamma.
  W5  Exact Psi of the centre of mass changes by > 0.01 nats between gamma = 0
      and gamma = 0.2, although its macro law is identical: Psi responds to
      micro correlations, not to the macro law.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

THETA = 0.1
GAMMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4)
SIZES = (10, 50)
TAUS = (1, 5)
T_SIM = 200_000


def laplacian(n: int) -> np.ndarray:
    lap = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    lap[0, 0] = lap[-1, -1] = 1.0
    return lap


def modes(n: int):
    lam, u = np.linalg.eigh(laplacian(n))
    lam[0] = 0.0
    u[:, 0] = 1.0 / np.sqrt(n)
    return lam, u


def coupling(n: int, gamma: float) -> np.ndarray:
    return (1.0 - THETA) * np.eye(n) - gamma * laplacian(n)


def share_exact(lam: float, gamma: float, tau: int) -> float:
    a, b = (1.0 - THETA - gamma * lam) ** tau, (1.0 - THETA) ** tau
    return (a - b) ** 2 / ((1.0 - a * a) + (a - b) ** 2)


def stationary_cov(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return (v / (1.0 - w ** 2)) @ v.T


def _mi(cov_xy: float, var_x: float, var_y: float) -> float:
    rho2 = cov_xy ** 2 / (var_x * var_y)
    return -0.5 * np.log1p(-min(rho2, 1 - 1e-15))


def psi_exact(n: int, gamma: float, u: np.ndarray, tau: int) -> float:
    """Psi = I(y_t; y_t+tau) - sum_j I(Delta_j(t); y_t+tau), exact Gaussian."""
    m = coupling(n, gamma)
    sigma = stationary_cov(m)
    lag = np.linalg.matrix_power(m, tau) @ sigma      # Cov(Delta(t+tau), Delta(t))
    var_y = u @ sigma @ u
    whole = _mi(u @ lag @ u, var_y, var_y)
    cross = u @ lag                                   # Cov(y(t+tau), Delta_j(t))
    parts = sum(_mi(cross[j], sigma[j, j], var_y) for j in range(n))
    return float(whole - parts)


def simulate(n: int, gamma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = coupling(n, gamma)
    x = rng.multivariate_normal(np.zeros(n), stationary_cov(m))
    noise = rng.standard_normal((T_SIM, n))
    out = np.empty((T_SIM, n))
    for t in range(T_SIM):
        x = m @ x + noise[t]
        out[t] = x
    return out


def share_sim(y: np.ndarray, tau: int) -> float:
    x, z = y[:-tau], y[tau:]
    half = len(x) // 2
    slope, icpt = np.polyfit(x[:half], z[:half], 1)
    mse_g = np.mean((z[half:] - slope * x[half:] - icpt) ** 2)
    mse_g0 = np.mean((z[half:] - (1.0 - THETA) ** tau * x[half:]) ** 2)
    return float(1.0 - mse_g / mse_g0)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    global T_SIM
    if quick:
        T_SIM = 40_000
    rows = {}
    for n in SIZES:
        lam, u = modes(n)
        picks = {"com": 0, "k=1": 1, f"k={n - 1}": n - 1}
        print(f"  N={n}: simulating {len(GAMMAS)} couplings ...", flush=True)
        for g in GAMMAS:
            traj = simulate(n, g, seed=23_000 + n + int(g * 1000)) @ u
            for name, k in picks.items():
                for tau in TAUS:
                    rows[(n, g, name, tau)] = (
                        share_exact(lam[k], g, tau), share_sim(traj[:, k], tau),
                        psi_exact(n, g, u[:, k], tau))
    return analyze(rows, out_dir)


def analyze(rows, out_dir: Path) -> str:
    lines = [f"Experiment 23: interaction share S vs Psi on coupled random walkers "
             f"(theta={THETA})", "",
             f"  {'N':>3}{'macro':>7}{'tau':>4}{'gamma':>7}{'S exact':>9}{'S sim':>8}"
             f"{'Psi':>8}"]
    for (n, g, name, tau), (se, ss, psi) in rows.items():
        lines.append(f"  {n:3d}{name:>7}{tau:4d}{g:7.2f}{se:9.4f}{ss:8.4f}{psi:8.3f}")

    w1_err = max(abs(ss) for (n, g, name, tau), (_, ss, _) in rows.items() if name == "com")
    w1 = w1_err <= 0.01
    w2 = all(rows[(n, 0.0, "com", tau)][2] > 0 for n in SIZES for tau in TAUS)
    w3 = True
    for n in SIZES:
        for tau in TAUS:
            s = [rows[(n, g, f"k={n - 1}", tau)][0] for g in GAMMAS]
            w3 &= s[0] == 0.0 and all(np.diff(s) > 0)
    w4_err = max(abs(se - ss) for se, ss, _ in rows.values())
    w4 = w4_err <= 0.02
    dpsi = min(abs(rows[(n, 0.2, "com", tau)][2] - rows[(n, 0.0, "com", tau)][2])
               for n in SIZES for tau in TAUS)
    w5 = dpsi > 0.01
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(w1)}  W1 simulated S(com) within 0.01 of 0 [max |S| = {w1_err:.4f}]",
              f"  {tag(w2)}  W2 Psi(com) > 0 at gamma = 0 "
              f"[{', '.join(f'N={n},tau={t}: {rows[(n, 0.0, 'com', t)][2]:.3f}' for n in SIZES for t in TAUS)}]",
              f"  {tag(w3)}  W3 S(k=N-1) zero at gamma=0, strictly increasing",
              f"  {tag(w4)}  W4 simulation matches closed form [max error {w4_err:.4f}]",
              f"  {tag(w5)}  W5 Psi(com) moves with gamma although its law does not "
              f"[min change {dpsi:.3f} nats]"]
    _plot(rows, out_dir / "e23_coupled_walkers.png")
    lines.append(f"\n  figure: {out_dir / 'e23_coupled_walkers.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(rows, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, tau = SIZES[0], TAUS[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    fine = np.linspace(0, GAMMAS[-1], 200)
    lam, _ = modes(n)
    for name, k, c in (("com", 0, "k"), ("k=1", 1, "tab:blue"), (f"k={n - 1}", n - 1, "tab:red")):
        ax[0].plot(fine, [share_exact(lam[k], g, tau) for g in fine], "-", color=c,
                   label=f"{name} exact")
        ax[0].plot(GAMMAS, [rows[(n, g, name, tau)][1] for g in GAMMAS], "o", color=c,
                   label=f"{name} simulated")
        ax[1].plot(GAMMAS, [rows[(n, g, name, tau)][2] for g in GAMMAS], "o-", color=c,
                   label=name)
    ax[0].set_title(f"Interaction share S (N={n}, tau={tau})")
    ax[1].set_title(f"Psi, exact Gaussian (N={n}, tau={tau})")
    ax[1].axhline(0, color="gray", lw=0.8)
    for a in ax:
        a.set_xlabel("coupling gamma")
        a.grid(alpha=0.2)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
