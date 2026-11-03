"""Experiment 45: the interaction-share spectrum. Finding, rather than choosing,
the collective variables that interactions shape (and those they cannot).

Until now we picked the collective variable (centre of mass, polarization,
...) and measured its interaction share S. Here the program is given only the
parts' recorded trajectories, the full model and the no-interaction twin, and
must find the collective variables itself.

Method. For a dictionary of observables psi(x) (the parts' states, or
functions of them), take the one-step prediction errors on the interacting
data of the full model, e = psi(x') - E_full[psi(x') | x], and of the twin,
e0 = psi(x') - E_twin[psi(x') | x]. For a collective variable y = w . psi,
    S(w) = 1 - w' Sigma w / w' Sigma0 w,  Sigma = E[e e'], Sigma0 = E[e0 e0'].
Its extremes solve the generalized eigenproblem Sigma w = lambda Sigma0 w,
S = 1 - lambda: every collective variable, ranked from most interaction-made
to interaction-blind, in one solve. Weights are fitted on the first half of
each run; every S reported is measured on the second half.

Why (Lean, InteractionShare.lean Part 4, `left_null_blind`): for linear
dynamics A = A0 + Delta, Sigma0 = Sigma + Delta C Delta' (C = state
covariance), so S(w) = 0 exactly when w' Delta = 0. The interaction-blind
variables are the left null space of the interaction operator. Newton's third
law (columns of Delta sum to 0) gives the centre of mass; one-way coupling
gives a weighted average with the left null vector pi as weights.

Systems (all simulated, so the twin is a real intervention):
  L1  e23's walkers on a path, N = 10, theta = 0.1, gamma = 0.2.
  L2  walkers coupled all-to-all, N = 10, theta = 0.1, gamma = 0.05.
  L3  walkers with one-way coupling on a random directed graph (ring plus
      10% extra edges, weights log-uniform 0.1-10), N = 10, theta = 0.1.
  N1  heterogeneous Kuramoto oscillators, N = 10, K = 1.5, noise 0.5,
      dictionary (theta_i, cos theta_i, sin theta_i), 30 observables.
  N2  agents in a periodic box, metric radius 4: alignment only, and
      alignment + attraction; dictionary = the N turns.

Resolution limit, known before running (exact, from e23's closed form): on
the path, the slow modes are barely touched by coupling (S_1 = 0.002,
S_2 = 0.007 at gamma = 0.2), so from 50 000 held-out steps they cannot be told
apart from the truly blind centre of mass. The method can only recover the
blind subspace up to such near-blind modes; L2 is the clean test.

Design facts (exact, from the models, before any simulation; see design()):
  L1: exact S spectrum 0.38 ... 0.018, 0.0017, 0 (two near-blind modes).
  L2: exact S = 0.229 for all 9 internal modes, 0 for the centre of mass.
  L3: spectral radius 0.90; |cos(pi, 1)| = 0.83, so the blind weighted
      average is far from the plain one; exact S of the plain centre of
      mass 0.18; next-most-blind exact S 0.055. (Edge weights were widened
      from 0.5-2 to 0.1-10 at the design stage because the first design gave
      S(centre of mass) = 0.03 and |cos| = 0.94: too mild a test.)

Predictions, fixed before running:
  Q1  L1: held-out spectrum (sorted S) within 0.02 of the exact spectrum,
      every direction.
  Q2  L1: the most interaction-blind discovered direction has exact S <= 0.01.
  Q3  L2: the most blind direction is the centre of mass (|cos| >= 0.99 with
      the uniform weights), held-out |S| <= 0.01, and the next held-out S is
      >= 0.1.
  Q4  L3: the most blind direction is the pi-weighted average (|cos| >= 0.99
      with pi), held-out |S| <= 0.01.
  Q5  L3: the plain centre of mass is NOT blind: held-out S within 0.02 of
      its exact value (which design() shows is >= 0.05).
  Q6  N1: the most blind direction is the mean phase (|cos| >= 0.95 with
      weights 1 on the theta_i block, 0 on cos/sin), held-out |S| <= 0.01.
  Q7  N2 alignment only: the most blind direction is the total turn
      (|cos| >= 0.99 with uniform weights), held-out |S| <= 0.01.
  Q8  N2 with attraction: the total turn is no longer blind, held-out
      S(total turn) >= 0.01.

Figures (results/e45_figures/) are drawn from the raw results after the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e23_coupled_walkers import laplacian

N = 10
THETA = 0.1
T_STEPS, BURN = 100_000, 1_000
SEED = 45


def gen_spectrum(e, e0):
    """Directions w (columns, unit norm) and S, sorted from most interaction-made to blind."""
    sig, sig0 = e.T @ e / len(e), e0.T @ e0 / len(e0)
    li = np.linalg.inv(np.linalg.cholesky(sig0))
    lam, v = np.linalg.eigh(li @ sig @ li.T)
    w = li.T @ v
    w /= np.linalg.norm(w, axis=0)
    return w, 1 - lam


def share(e, e0, w):
    return 1 - ((e @ w) ** 2).mean(0) / ((e0 @ w) ** 2).mean(0)


def cosine(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def stationary_cov(a):
    n = len(a)
    return np.linalg.solve(np.eye(n * n) - np.kron(a, a), np.eye(n).ravel()).reshape(n, n)


def exact_linear(a, a0):
    d = a - a0
    sig0 = np.eye(len(a)) + d @ stationary_cov(a) @ d.T
    return np.eye(len(a)), sig0


def exact_share(sig, sig0, w):
    return float(1 - (w @ sig @ w) / (w @ sig0 @ w))


def digraph(rng):
    w = np.zeros((N, N))
    for i in range(N):
        w[i, (i + 1) % N] = np.exp(rng.uniform(np.log(0.1), np.log(10.0)))
    extra = rng.random((N, N)) < 0.10
    np.fill_diagonal(extra, False)
    w[extra] = np.exp(rng.uniform(np.log(0.1), np.log(10.0), extra.sum()))
    lap = np.diag(w.sum(1)) - w
    gamma = 0.9 / w.sum(1).max()
    vals, vecs = np.linalg.eig(lap.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals))])
    pi = pi / pi.sum()
    return lap, gamma, pi


def linear_systems():
    rng = np.random.default_rng(SEED)
    lap_d, gamma_d, pi = digraph(rng)
    full = np.ones((N, N)) - N * np.eye(N)
    return {
        "L1": ((1 - THETA) * np.eye(N) - 0.2 * laplacian(N), None),
        "L2": ((1 - THETA) * np.eye(N) + 0.05 * full, None),
        "L3": ((1 - THETA) * np.eye(N) - gamma_d * lap_d, pi),
    }


def design():
    a0 = (1 - THETA) * np.eye(N)
    out = {}
    for name, (a, pi) in linear_systems().items():
        sig, sig0 = exact_linear(a, a0)
        w, s = gen_spectrum_exact(sig, sig0)
        com = np.ones(N) / np.sqrt(N)
        out[name] = {"radius": float(max(abs(np.linalg.eigvals(a)))), "S_exact": s.tolist(),
                     "S_com": exact_share(sig, sig0, com)}
        if pi is not None:
            out[name]["cos_pi_1"] = cosine(pi, np.ones(N))
            out[name]["pi"] = pi.tolist()
    return out


def gen_spectrum_exact(sig, sig0):
    li = np.linalg.inv(np.linalg.cholesky(sig0))
    lam, v = np.linalg.eigh(li @ sig @ li.T)
    w = li.T @ v
    return w / np.linalg.norm(w, axis=0), 1 - lam


def run_linear(name, a, pi, rng):
    a0 = (1 - THETA) * np.eye(N)
    x = np.zeros((T_STEPS + BURN + 1, N))
    noise = rng.standard_normal(x.shape)
    for t in range(len(x) - 1):
        x[t + 1] = a @ x[t] + noise[t]
    x = x[BURN:]
    e, e0 = x[1:] - x[:-1] @ a.T, x[1:] - x[:-1] @ a0.T
    half = len(e) // 2
    w, s_fit = gen_spectrum(e[:half], e0[:half])
    s_test = share(e[half:], e0[half:], w)
    sig, sig0 = exact_linear(a, a0)
    _, s_exact = gen_spectrum_exact(sig, sig0)
    com = np.ones(N) / np.sqrt(N)
    blind = w[:, -1]
    out = {"name": name, "w": w.tolist(), "S_fit": s_fit.tolist(), "S_test": s_test.tolist(),
           "S_exact": s_exact.tolist(), "S_exact_of_w": [exact_share(sig, sig0, w[:, k]) for k in range(N)],
           "cos_com": cosine(blind, com), "S_com_test": float(share(e[half:], e0[half:], com[:, None])[0]),
           "S_com_exact": exact_share(sig, sig0, com), "A": a.tolist()}
    if pi is not None:
        out.update({"pi": pi.tolist(), "cos_pi": cosine(blind, pi), "cos_pi_1": cosine(pi, np.ones(N))})
    return out


def run_kuramoto(rng):
    k, sigma, dt = 1.5, 0.5, 0.05
    omega = rng.normal(0.0, 0.5, N)
    s2 = sigma ** 2 * dt
    th = np.zeros((T_STEPS + BURN + 1, N))
    th[0] = rng.uniform(-np.pi, np.pi, N)
    noise = rng.standard_normal(th.shape) * np.sqrt(s2)
    for t in range(len(th) - 1):
        c = (k / N) * np.sin(th[t][None, :] - th[t][:, None]).sum(1)
        th[t + 1] = th[t] + dt * (omega + c) + noise[t]
    th = th[BURN:]
    prev, nxt = th[:-1], th[1:]
    coup = (k / N) * np.sin(prev[:, None, :] - prev[:, :, None]).sum(2)
    damp = np.exp(-s2 / 2)

    def expect(mu):
        return np.concatenate([mu, np.cos(mu) * damp, np.sin(mu) * damp], 1)

    psi = np.concatenate([nxt, np.cos(nxt), np.sin(nxt)], 1)
    e = psi - expect(prev + dt * (omega + coup))
    e0 = psi - expect(prev + dt * omega)
    half = len(e) // 2
    w, s_fit = gen_spectrum(e[:half], e0[:half])
    s_test = share(e[half:], e0[half:], w)
    target = np.concatenate([np.ones(N), np.zeros(2 * N)])
    r = np.abs(np.exp(1j * th).mean(1))
    return {"name": "N1", "w": w.tolist(), "S_fit": s_fit.tolist(), "S_test": s_test.tolist(),
            "cos_target": cosine(w[:, -1], target),
            "S_target_test": float(share(e[half:], e0[half:], target[:, None])[0]),
            "order_param_mean": float(r.mean()), "omega": omega.tolist()}


def run_agents(rng, align, attract):
    n, box, radius, speed, sigma, dt = 12, 15.0, 4.0, 2.0, 0.4, 0.1
    steps = T_STEPS + BURN
    p = rng.uniform(0, box, (n, 2))
    th = rng.uniform(-np.pi, np.pi, n)
    e, e0, snaps = [], [], []
    for t in range(steps):
        d = p[None, :, :] - p[:, None, :]
        d -= box * np.round(d / box)
        dist = np.hypot(d[..., 0], d[..., 1])
        nb = (dist < radius) & ~np.eye(n, dtype=bool)
        bearing = np.arctan2(d[..., 1], d[..., 0])
        coup = align * (np.sin(th[None, :] - th[:, None]) * nb).sum(1) \
            + attract * (np.sin(bearing - th[:, None]) * nb).sum(1)
        eta = rng.standard_normal(n) * sigma * np.sqrt(dt)
        turn = dt * coup + eta
        if t >= BURN:
            e.append(eta)
            e0.append(turn)
            if t % 20_000 == 0 or t == steps - 1:
                snaps.append({"p": p.tolist(), "th": th.tolist()})
        th = (th + turn + np.pi) % (2 * np.pi) - np.pi
        p = (p + speed * dt * np.stack([np.cos(th), np.sin(th)], 1)) % box
    e, e0 = np.array(e), np.array(e0)
    half = len(e) // 2
    w, s_fit = gen_spectrum(e[:half], e0[:half])
    s_test = share(e[half:], e0[half:], w)
    ones = np.ones(n)
    return {"name": "N2-" + ("align" if attract == 0 else "attract"), "w": w.tolist(), "S_fit": s_fit.tolist(),
            "S_test": s_test.tolist(), "cos_total": cosine(w[:, -1], ones),
            "S_total_test": float(share(e[half:], e0[half:], ones[:, None])[0]), "snaps": snaps,
            "box": box, "radius": radius}


def predictions(res):
    l1, l2, l3, n1 = res["L1"], res["L2"], res["L3"], res["N1"]
    al, at = res["N2-align"], res["N2-attract"]
    gap1 = np.max(np.abs(np.sort(l1["S_test"]) - np.sort(l1["S_exact"])))
    return [
        ("Q1", "path walkers: held-out spectrum within 0.02 of exact", gap1 <= 0.02, f"max gap {gap1:.4f}"),
        ("Q2", "path walkers: most blind direction has exact S <= 0.01", l1["S_exact_of_w"][-1] <= 0.01,
         f"exact S {l1['S_exact_of_w'][-1]:.4f}, |cos| with centre of mass {l1['cos_com']:.3f}"),
        ("Q3", "all-to-all walkers: blind direction = centre of mass, gap >= 0.1",
         l2["cos_com"] >= 0.99 and abs(l2["S_test"][-1]) <= 0.01 and l2["S_test"][-2] >= 0.1,
         f"|cos| {l2['cos_com']:.4f}, S {l2['S_test'][-1]:+.4f}, next S {l2['S_test'][-2]:.3f}"),
        ("Q4", "directed graph: blind direction = pi-weighted average",
         l3["cos_pi"] >= 0.99 and abs(l3["S_test"][-1]) <= 0.01,
         f"|cos| with pi {l3['cos_pi']:.4f} (with uniform {l3['cos_com']:.3f}), S {l3['S_test'][-1]:+.4f}"),
        ("Q5", "directed graph: plain centre of mass not blind, matches exact",
         abs(l3["S_com_test"] - l3["S_com_exact"]) <= 0.02 and l3["S_com_exact"] >= 0.05,
         f"held-out {l3['S_com_test']:.4f}, exact {l3['S_com_exact']:.4f}"),
        ("Q6", "Kuramoto: blind direction = mean phase",
         n1["cos_target"] >= 0.95 and abs(n1["S_test"][-1]) <= 0.01,
         f"|cos| {n1['cos_target']:.4f}, S {n1['S_test'][-1]:+.4f}"),
        ("Q7", "agents, alignment: blind direction = total turn",
         al["cos_total"] >= 0.99 and abs(al["S_test"][-1]) <= 0.01,
         f"|cos| {al['cos_total']:.4f}, S {al['S_test'][-1]:+.4f}"),
        ("Q8", "agents, + attraction: total turn not blind (S >= 0.01)", at["S_total_test"] >= 0.01,
         f"S(total turn) {at['S_total_test']:.4f}; most blind |cos| {at['cos_total']:.3f}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 45: the interaction-share spectrum (collective variables found, not chosen)", ""]
    for key in ("L1", "L2", "L3"):
        r = res[key]
        lines.append(f"  {key}: held-out S (most interaction-made -> blind): "
                     + " ".join(f"{x:.3f}" for x in r["S_test"]))
        lines.append(f"      exact S:                                  "
                     + " ".join(f"{x:.3f}" for x in r["S_exact"]))
        lines.append(f"      most blind: |cos| with centre of mass {r['cos_com']:.4f}"
                     + (f", with pi {r['cos_pi']:.4f}" if "cos_pi" in r else "")
                     + f"; S(centre of mass) held-out {r['S_com_test']:+.4f}, exact {r['S_com_exact']:+.4f}")
    n1 = res["N1"]
    lines.append(f"  N1 Kuramoto (order parameter {n1['order_param_mean']:.2f}): smallest five held-out S "
                 + " ".join(f"{x:+.4f}" for x in n1["S_test"][-5:])
                 + f"; most blind |cos| with mean phase {n1['cos_target']:.4f}; top S {n1['S_test'][0]:.3f}")
    for key in ("N2-align", "N2-attract"):
        r = res[key]
        lines.append(f"  {key}: held-out S " + " ".join(f"{x:.3f}" for x in r["S_test"])
                     + f"\n      most blind |cos| with total turn {r['cos_total']:.4f}; "
                       f"S(total turn) {r['S_total_test']:+.4f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    rng = np.random.default_rng(SEED + 1)
    res = {name: run_linear(name, a, pi, rng) for name, (a, pi) in linear_systems().items()}
    res["N1"] = run_kuramoto(rng)
    res["N2-align"] = run_agents(rng, 1.0, 0.0)
    res["N2-attract"] = run_agents(rng, 1.0, 0.8)
    (out_dir / "e45_interaction_spectrum_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e45_figures import make_figures
    make_figures(res, out_dir / "e45_figures", text)
    return text


if __name__ == "__main__":
    main()
