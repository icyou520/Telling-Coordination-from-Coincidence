"""Experiment 46: the interaction-share spectrum with an ESTIMATED twin, then
real fish.

e45 found the interaction-blind collective variables when the full model and
the no-interaction twin were known. On data both must be estimated. e24
showed the trap: a twin fitted from each part's own past absorbs the
coupling. The convention that worked there: interactions are functions of
the parts' RELATIVE states (differences), so they vanish when all parts are
alike; the own part of a fitted full model is then the twin.

Estimators (data only; no graph, no true coefficients). Each run is split in
thirds: models fitted on the first, spectrum weights on the second, every S
reported on the last.
  Linear (e45's L3, one-way coupling on a random digraph): fit the full VAR
      x' = A x by least squares; the relative-state convention
      x_i' = alpha_i x_i + sum_j k_ij (x_j - x_i) gives the twin
      alpha_i = row sum of A. Naive control (as in e24): each walker
      regressed on its own past only.
  Kuramoto (e45's N1): per oscillator, dtheta_i = w_i dt + sum_j
      k_ij sin(theta_j - theta_i); twin = w_i only.
  Agents (e45's N2): turn_i = b_i + sum over distance bins (0-2, 2-4, 4-6)
      of a_bin * sum sin(theta_j - theta_i) + c_bin * sum sin(bearing_ij -
      theta_i), shared across agents; twin = b_i.
  Fish (Puy et al., 8-fish recordings, real and virtual): e44's model with
      rule k6-AC fitted on the first 70% (per-fish turning and speed maps,
      shared social terms); twin = the same fit's own part. Dictionary: the
      8 turns and the 8 speed changes (16 observables). Spectrum weights on
      the next 15%, S on the last 15%.

Predictions, fixed before running:
  R1  Linear, data-only twin, T = 100 000: most blind direction = the pi-weighted
      average (|cos| >= 0.99 with the true pi), held-out |S| <= 0.01.
  R2  Linear, short data (T = 5 000, 20 replicates): median |cos| with pi >= 0.95.
  R3  Linear, naive twin: misidentifies the blind variable, |cos| with pi < 0.95.
  R4  Kuramoto, data-only twin: most blind = mean phase (|cos| >= 0.95 with
      the theta-block uniform target), held-out |S| <= 0.01.
  R5  Agents, data-only twin: alignment only -> most blind = total turn
      (|cos| >= 0.99), |S| <= 0.01; with attraction -> S(total turn) >= 0.01.
  R6  Fish control: in the 6 virtual groups (never met), the largest held-out
      S is <= 0.03 in >= 5 of 6.
  R7  Fish: in the 6 real groups, the largest held-out S is >= 0.05 in >= 5 of 6,
      and exceeds its virtual pair's in 6 of 6.
Exploratory: what the most interaction-made and most blind collective
variables of real schools are (weights over the 8 turns and 8 speed changes),
and S of the total turn (not protected by the Lean theorem, since k6 is a
one-sided neighbourhood).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e45_interaction_spectrum import N, cosine, gen_spectrum, linear_systems, share

T_LONG, T_SHORT, REPS, BURN = 100_000, 5_000, 20, 1_000
SEED = 46


def thirds(n):
    a, b = n // 3, 2 * n // 3
    return slice(0, a), slice(a, b), slice(b, n)


def spectrum_split(e, e0, fit, test):
    w, _ = gen_spectrum(e[fit], e0[fit])
    return w, share(e[test], e0[test], w)


# ---------------------------------------------------------------- linear
def simulate_linear(a, steps, rng):
    x = np.zeros((steps + BURN + 1, N))
    noise = rng.standard_normal(x.shape)
    for t in range(len(x) - 1):
        x[t + 1] = a @ x[t] + noise[t]
    return x[BURN:]


def linear_estimated(x, naive=False):
    prev, nxt = x[:-1], x[1:]
    m, s, t = thirds(len(prev))
    a_hat = np.linalg.lstsq(prev[m], nxt[m], rcond=None)[0].T
    if naive:
        twin = np.diag([np.dot(prev[m, i], nxt[m, i]) / np.dot(prev[m, i], prev[m, i]) for i in range(N)])
    else:
        twin = np.diag(a_hat.sum(1))
    e, e0 = nxt - prev @ a_hat.T, nxt - prev @ twin.T
    return spectrum_split(e, e0, s, t)


def run_linear(rng):
    a, pi = linear_systems()["L3"]
    x = simulate_linear(a, T_LONG, rng)
    w, s = linear_estimated(x)
    wn, sn = linear_estimated(x, naive=True)
    short = []
    for _ in range(REPS):
        ws, ss = linear_estimated(simulate_linear(a, T_SHORT, rng))
        short.append(cosine(ws[:, -1], pi))
    return {"pi": pi.tolist(), "w": w.tolist(), "S_test": s.tolist(), "cos_pi": cosine(w[:, -1], pi),
            "w_naive": wn.tolist(), "S_naive": sn.tolist(), "cos_pi_naive": cosine(wn[:, -1], pi),
            "cos_pi_short": short}


# ---------------------------------------------------------------- Kuramoto
def run_kuramoto(rng):
    k, sigma, dt = 1.5, 0.5, 0.05
    omega = rng.normal(0.0, 0.5, N)
    s2 = sigma ** 2 * dt
    th = np.zeros((T_LONG + BURN + 1, N))
    th[0] = rng.uniform(-np.pi, np.pi, N)
    noise = rng.standard_normal(th.shape) * np.sqrt(s2)
    for t in range(len(th) - 1):
        c = (k / N) * np.sin(th[t][None, :] - th[t][:, None]).sum(1)
        th[t + 1] = th[t] + dt * (omega + c) + noise[t]
    th = th[BURN:]
    prev, nxt = th[:-1], th[1:]
    m, s, t = thirds(len(prev))
    rel = np.sin(prev[:, None, :] - prev[:, :, None])
    dth = nxt - prev
    drift_full, drift_twin = np.zeros_like(prev), np.zeros_like(prev)
    for i in range(N):
        others = [j for j in range(N) if j != i]
        x = np.column_stack([np.ones(len(prev)), rel[:, i, others]])
        coef = np.linalg.lstsq(x[m], dth[m, i], rcond=None)[0]
        drift_full[:, i] = x @ coef
        drift_twin[:, i] = coef[0]
    resid = dth[m] - drift_full[m]
    damp = np.exp(-resid.var() / 2)

    def expect(mu):
        return np.concatenate([mu, np.cos(mu) * damp, np.sin(mu) * damp], 1)

    psi = np.concatenate([nxt, np.cos(nxt), np.sin(nxt)], 1)
    e, e0 = psi - expect(prev + drift_full), psi - expect(prev + drift_twin)
    w, st = spectrum_split(e, e0, s, t)
    target = np.concatenate([np.ones(N), np.zeros(2 * N)])
    return {"w": w.tolist(), "S_test": st.tolist(), "cos_target": cosine(w[:, -1], target)}


# ---------------------------------------------------------------- agents
BINS = (0.0, 2.0, 4.0, 6.0)


def run_agents(rng, align, attract):
    n, box, radius, speed, sigma, dt = 12, 15.0, 4.0, 2.0, 0.4, 0.1
    p = rng.uniform(0, box, (n, 2))
    th = rng.uniform(-np.pi, np.pi, n)
    feats, turns = [], []
    for t in range(T_LONG + BURN):
        d = p[None, :, :] - p[:, None, :]
        d -= box * np.round(d / box)
        dist = np.hypot(d[..., 0], d[..., 1])
        np.fill_diagonal(dist, np.inf)
        nb = dist < radius
        bearing = np.arctan2(d[..., 1], d[..., 0])
        sal, sat = np.sin(th[None, :] - th[:, None]), np.sin(bearing - th[:, None])
        coup = align * (sal * nb).sum(1) + attract * (sat * nb).sum(1)
        turn = dt * coup + rng.standard_normal(n) * sigma * np.sqrt(dt)
        if t >= BURN:
            f = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                b = (dist >= lo) & (dist < hi)
                f += [(sal * b).sum(1), (sat * b).sum(1)]
            feats.append(np.stack(f, 1))
            turns.append(turn)
        th = (th + turn + np.pi) % (2 * np.pi) - np.pi
        p = (p + speed * dt * np.stack([np.cos(th), np.sin(th)], 1)) % box
    f, y = np.array(feats), np.array(turns)
    m, s, t = thirds(len(y))
    x = np.concatenate([np.repeat(np.eye(n)[None], len(y), 0), f], 2)
    coef = np.linalg.lstsq(x[m].reshape(-1, x.shape[-1]), y[m].ravel(), rcond=None)[0]
    full, twin = x @ coef, x[..., :n] @ coef[:n]
    e, e0 = y - full, y - twin
    w, st = spectrum_split(e, e0, s, t)
    ones = np.ones(n)
    return {"w": w.tolist(), "S_test": st.tolist(), "cos_total": cosine(w[:, -1], ones),
            "S_total_test": float(share(e[t], e0[t], ones[:, None])[0]), "social_coef": coef[n:].tolist()}


# ---------------------------------------------------------------- fish
def run_fish(args):
    rec, kind = args
    import experiments.e44_elimination_speed_heading as m
    from experiments.e31_real_fish import kinematics, virtual
    from experiments.fish_data import load
    base = m.base
    pos = load(8, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    sp = np.where(sp > 100.0, 0.0, sp)
    T = len(th)
    cut = int(0.7 * T)
    h = "k6-AC"
    fits, own, feats, _, (l1, l2, ok_in, dvp) = m.fit_all(p, th, sp, cut, [h])
    ct, bt, _, cv, bv, _ = fits[h]
    own_v = m.own_speed_x(p, sp, dvp, th)
    zt, zv = base.social_turn(feats, h)[:-1], base.social_speed(feats, h)[:-1]
    twin_t, twin_v = (own[:-1] * ct).sum(-1), (own_v[:-1] * cv).sum(-1)
    full_t, full_v = twin_t + zt @ bt, twin_v + zv @ bv
    yt, yv = l1[1:], dvp[1:]
    rows = np.nonzero(ok_in[1:].all(1) & (np.arange(T - 1) >= cut))[0]
    e = np.concatenate([yt - full_t, yv - full_v], 1)[rows]
    e0 = np.concatenate([yt - twin_t, yv - twin_v], 1)[rows]
    half = len(rows) // 2
    w, st = spectrum_split(e, e0, slice(0, half), slice(half, None))
    tot = np.concatenate([np.ones(8), np.zeros(8)])
    spd = np.concatenate([np.zeros(8), np.ones(8)])
    return {"rec": rec, "kind": kind, "w": w.tolist(), "S_test": st.tolist(), "rows": len(rows),
            "S_total_turn": float(share(e[half:], e0[half:], tot[:, None])[0]),
            "S_total_speed": float(share(e[half:], e0[half:], spd[:, None])[0]),
            "cos_total_turn_blind": cosine(w[:, -1], tot), "social_turn": bt.tolist(),
            "social_speed": bv.tolist()}


# ---------------------------------------------------------------- analysis
def predictions(res):
    lin, ku, al, at = res["linear"], res["kuramoto"], res["agents_align"], res["agents_attract"]
    real = sorted([r for r in res["fish"] if r["kind"] == "real"], key=lambda r: r["rec"])
    virt = sorted([r for r in res["fish"] if r["kind"] == "virtual"], key=lambda r: r["rec"])
    med = float(np.median(lin["cos_pi_short"]))
    r6 = sum(max(v["S_test"]) <= 0.03 for v in virt)
    r7a = sum(max(r["S_test"]) >= 0.05 for r in real)
    r7b = sum(max(r["S_test"]) > max(v["S_test"]) for r, v in zip(real, virt))
    return [
        ("R1", "linear, data-only twin: blind = pi-weighted average",
         lin["cos_pi"] >= 0.99 and abs(lin["S_test"][-1]) <= 0.01,
         f"|cos| {lin['cos_pi']:.4f}, S {lin['S_test'][-1]:+.4f}"),
        ("R2", "linear, short data (T = 5000): median |cos| >= 0.95", med >= 0.95,
         f"median {med:.3f}, min {min(lin['cos_pi_short']):.3f}"),
        ("R3", "linear, naive twin misidentifies (|cos| < 0.95)", lin["cos_pi_naive"] < 0.95,
         f"|cos| {lin['cos_pi_naive']:.3f}, its S {lin['S_naive'][-1]:+.3f}"),
        ("R4", "Kuramoto, data-only twin: blind = mean phase",
         ku["cos_target"] >= 0.95 and abs(ku["S_test"][-1]) <= 0.01,
         f"|cos| {ku['cos_target']:.4f}, S {ku['S_test'][-1]:+.4f}"),
        ("R5", "agents, data-only twin: total turn blind / not blind",
         al["cos_total"] >= 0.99 and abs(al["S_test"][-1]) <= 0.01 and at["S_total_test"] >= 0.01,
         f"align |cos| {al['cos_total']:.4f}, S {al['S_test'][-1]:+.4f}; attract S(total) "
         f"{at['S_total_test']:.3f}"),
        ("R6", "fish virtual: largest S <= 0.03 (>= 5/6)", r6 >= 5,
         "max S " + ", ".join(f"{max(v['S_test']):.3f}" for v in virt)),
        ("R7", "fish real: largest S >= 0.05 (>= 5/6), > virtual pair (6/6)", r7a >= 5 and r7b == 6,
         "max S " + ", ".join(f"{max(r['S_test']):.3f}" for r in real) + f"; above pair {r7b}/6"),
    ]


def analyze(res) -> str:
    lin = res["linear"]
    lines = ["Experiment 46: interaction-share spectrum with estimated twins, and real fish", "",
             f"  Linear (one-way coupling), data-only twin: held-out S " + " ".join(f"{x:.3f}" for x in lin["S_test"]),
             f"      naive twin:                           " + " ".join(f"{x:.3f}" for x in lin["S_naive"]),
             f"  Kuramoto, data-only: top S {res['kuramoto']['S_test'][0]:.3f}, smallest "
             + " ".join(f"{x:+.4f}" for x in res["kuramoto"]["S_test"][-3:]),
             f"  Agents align: S " + " ".join(f"{x:.3f}" for x in res["agents_align"]["S_test"]),
             f"  Agents attract: S " + " ".join(f"{x:.3f}" for x in res["agents_attract"]["S_test"]),
             "", "  Fish (8-fish recordings): held-out S, top 4 and most blind; S(total turn); S(total speed change)"]
    for r in sorted(res["fish"], key=lambda r: (r["kind"], r["rec"])):
        lines.append(f"    {r['kind']:7} #{r['rec']}  top " + " ".join(f"{x:.3f}" for x in r["S_test"][:4])
                     + f"  blind {r['S_test'][-1]:+.3f}  total turn {r['S_total_turn']:+.3f}"
                       f"  total speed {r['S_total_speed']:+.3f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    rng = np.random.default_rng(SEED)
    with ProcessPoolExecutor(max_workers=8) as pool:
        fish = pool.map(run_fish, [(r, k) for r in range(6) for k in ("real", "virtual")])
        res = {"linear": run_linear(rng), "kuramoto": run_kuramoto(rng),
               "agents_align": run_agents(rng, 1.0, 0.0), "agents_attract": run_agents(rng, 1.0, 0.8)}
        res["fish"] = list(fish)
    (out_dir / "e46_spectrum_estimated_twin_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e46_figures import make_figures
    make_figures(res, out_dir / "e46_figures", text)
    return text


if __name__ == "__main__":
    main()
