"""Experiment 24: can the no-interaction twin be estimated from data alone?

In e23 the twin's law was known exactly. Real data only gives trajectories
of the interacting parts. Three ways to estimate the twin, each fitted pooled
over walkers on the first half of a run:

  fitted   the mechanistic model  Delta_i' = alpha Delta_i
           + kappa sum_{j ~ i} (Delta_j - Delta_i) + noise,
           with interactions written as functions of *relative* states
           on the true graph; the twin keeps alpha and drops kappa.
  naive    each walker regressed on its own past only (no model of
           interaction); its slope is used as the twin.
  meanfld  interactions misspecified as all-to-all,
           Delta_i' = alpha Delta_i + kappa (V - Delta_i); twin keeps alpha.

S is then estimated on the second half exactly as in e23, with the fitted
twin slope b^tau in place of the true (1 - theta)^tau.

Hypothesis: the twin is identifiable only given a decomposition of the
dynamics into own-state terms and interaction terms. Without one (naive),
the twin absorbs the coupling and S is wrong.

Predictions, fixed before running:

  T1  Fitted twin, T = 200 000: S within 0.02 of the closed form for every
      mode and coupling, at theta = 0.1 and at theta = 0 (Liardi et al.'s
      non-stationary case, where the centre of mass is a free random walk).
  T2  Naive twin gives a false positive: S(centre of mass) >= 0.05 at
      gamma = 0.2, theta = 0.1, N = 10, where the true value is 0.
  T3  Fitted twin with short data (T = 2000, 30 replicates, theta = 0.1):
      mean |S - exact| <= 0.03 over modes and couplings, and
      |S(centre of mass)| <= 0.02 in >= 90 % of replicates at every gamma.

Exploratory (no prediction): the misspecified mean-field twin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.e23_coupled_walkers import GAMMAS, laplacian, modes

N = 10
THETAS = (0.1, 0.0)
TAU = 1
T_LONG = 200_000
T_SHORT = 2_000
REPLICATES = 30
BURN_IN = 1_000


def share_exact(lam: float, gamma: float, theta: float, tau: int = TAU) -> float:
    a, b = (1.0 - theta - gamma * lam) ** tau, (1.0 - theta) ** tau
    d = (a - b) ** 2
    return 0.0 if d == 0.0 else d / ((1.0 - a * a) + d)


def simulate(gamma: float, theta: float, steps: int, rng) -> np.ndarray:
    m = (1.0 - theta) * np.eye(N) - gamma * laplacian(N)
    noise = rng.standard_normal((BURN_IN + steps, N))
    x = np.zeros(N)
    out = np.empty((steps, N))
    for t in range(BURN_IN + steps):
        x = m @ x + noise[t]
        if t >= BURN_IN:
            out[t - BURN_IN] = x
    return out


def twin_slopes(x: np.ndarray) -> dict[str, float]:
    """Own-state coefficient alpha under each interaction model."""
    now, nxt = x[:-1], x[1:]
    y = nxt.ravel()
    own = now.ravel()
    rel_graph = -(now @ laplacian(N)).ravel()            # sum_{j~i} (x_j - x_i)
    rel_mean = (now.mean(1, keepdims=True) - now).ravel()  # V - x_i
    fit = lambda cols: np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0][0]
    return {"fitted": fit([own, rel_graph]), "naive": fit([own]), "meanfld": fit([own, rel_mean])}


def share_heldout(y: np.ndarray, b: float, tau: int = TAU) -> float:
    x, z = y[:-tau], y[tau:]
    half = len(x) // 2
    slope, icpt = np.polyfit(x[:half], z[:half], 1)
    mse_g = np.mean((z[half:] - slope * x[half:] - icpt) ** 2)
    mse_g0 = np.mean((z[half:] - b ** tau * x[half:]) ** 2)
    return float(1.0 - mse_g / mse_g0)


def estimate(x: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
    slopes = twin_slopes(x[: len(x) // 2])
    y = x @ u
    return {k: np.array([share_heldout(y[:, j], b) for j in range(u.shape[1])])
            for k, b in slopes.items()} | {"slopes": slopes}


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    lam, u = modes(N)
    picks = [0, 1, N - 1]
    t_long = 40_000 if quick else T_LONG
    reps = 5 if quick else REPLICATES
    long, short = {}, {}
    for theta in THETAS:
        print(f"  theta={theta}: long runs ...", flush=True)
        for g in GAMMAS:
            rng = np.random.default_rng(24_000 + int(1000 * g) + int(100 * theta))
            est = estimate(simulate(g, theta, t_long, rng), u[:, picks])
            long[(theta, g)] = est | {"exact": np.array([share_exact(lam[k], g, theta)
                                                         for k in picks])}
    print(f"  short runs ({reps} replicates x {len(GAMMAS)} couplings) ...", flush=True)
    for g in GAMMAS:
        rng = np.random.default_rng(24_500 + int(1000 * g))
        short[g] = np.array([estimate(simulate(g, 0.1, T_SHORT, rng), u[:, picks])["fitted"]
                             for _ in range(reps)])
    return analyze(long, short, lam, picks, out_dir)


def analyze(long, short, lam, picks, out_dir: Path) -> str:
    names = ["com", "k=1", f"k={N - 1}"]
    lines = [f"Experiment 24: estimated no-interaction twin (N={N}, tau={TAU})", "",
             f"  {'theta':>5}{'gamma':>6}{'mode':>6}{'exact':>8}{'fitted':>8}{'naive':>8}"
             f"{'meanfld':>9}   twin slopes (fitted / naive / meanfld)"]
    for (theta, g), e in long.items():
        s = e["slopes"]
        for i, nm in enumerate(names):
            tail = (f"   {s['fitted']:.3f} / {s['naive']:.3f} / {s['meanfld']:.3f}"
                    if i == 0 else "")
            lines.append(f"  {theta:5.1f}{g:6.2f}{nm:>6}{e['exact'][i]:8.4f}{e['fitted'][i]:8.4f}"
                         f"{e['naive'][i]:8.4f}{e['meanfld'][i]:9.4f}{tail}")

    t1_err = max(float(np.max(np.abs(e["fitted"] - e["exact"]))) for e in long.values())
    t1 = t1_err <= 0.02
    t2_val = float(long[(0.1, 0.2)]["naive"][0])
    t2 = t2_val >= 0.05
    exact_short = {g: long[(0.1, g)]["exact"] for g in GAMMAS}
    mae = float(np.mean([np.abs(short[g] - exact_short[g]).mean() for g in GAMMAS]))
    com_ok = min(float(np.mean(np.abs(short[g][:, 0]) <= 0.02)) for g in GAMMAS)
    t3 = mae <= 0.03 and com_ok >= 0.9
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", f"  Short data (T={T_SHORT}, theta=0.1), fitted twin, mean +- sd over replicates:"]
    for g in GAMMAS:
        cells = "  ".join(f"{nm} {short[g][:, i].mean():+.3f}+-{short[g][:, i].std():.3f}"
                          f" (exact {exact_short[g][i]:.3f})" for i, nm in enumerate(names))
        lines.append(f"    gamma {g:4.2f}: {cells}")
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(t1)}  T1 fitted twin within 0.02 of exact, theta 0.1 and 0 "
              f"[max error {t1_err:.4f}]",
              f"  {tag(t2)}  T2 naive twin false positive S(com) >= 0.05 at gamma 0.2 "
              f"[{t2_val:.3f}]",
              f"  {tag(t3)}  T3 short data: mean |error| {mae:.4f} (<= 0.03), "
              f"|S(com)| <= 0.02 in {100 * com_ok:.0f}% of replicates at worst gamma (>= 90%)"]
    _plot(long, names, out_dir / "e24_estimated_twin.png")
    lines.append(f"\n  figure: {out_dir / 'e24_estimated_twin.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(long, names, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    colors = {"exact": "k", "fitted": "tab:green", "naive": "tab:red", "meanfld": "tab:orange"}
    for i, nm in enumerate(names):
        for key, c in colors.items():
            ax[i].plot(GAMMAS, [long[(0.1, g)][key][i] for g in GAMMAS],
                       "-" if key == "exact" else "o--", color=c, label=f"{key} twin"
                       if key != "exact" else "exact")
        ax[i].set_title(f"{nm} (theta=0.1)")
        ax[i].set_xlabel("coupling gamma")
        ax[i].grid(alpha=0.2)
    ax[0].set_ylabel("interaction share S")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
