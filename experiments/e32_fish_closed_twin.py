"""Experiment 32: real fish schools with a persistent twin and a closed macro state.

e31 failed 3 of 4 predictions for two diagnosed reasons: the twin drew
independent turning noise although real turning is persistent (lag-0.1 s
autocorrelation 0.47), and polarization alone is not closed for the twin in
a walled tank. This experiment fixes both before looking at any S value.

Changes from e31 (everything else as there: 0.1 s sampling, tau = 1 s,
4 nearest neighbours, first 70 % train / last 30 % test, virtual groups by
time shift):

  Twin       heading change = bias + wall terms + rho1 * previous turn
             + rho2 * turn before that + resampled residual. The lags are the
             fish's own state, so they stay in the twin; alignment, attraction
             and repulsion are dropped. Lag turns are set to 0 where the fish
             was slower than 1 cm/s.
  Macro      the target is always polarization phi(t + 1 s). The macro state
             it is predicted from is chosen from this ordered list:
               M1  phi
               M2  phi, W = mean exp(-d_wall / 15 cm)
               M3  M2 + H = mean cos(heading - inward wall normal)
                    * exp(-d_wall / 15 cm)
               M4  M3 + Q = mean |previous turn|
             G is a full cubic polynomial in the (standardized) macro state.
  Selection  the chosen macro state is the first in the list whose twin
             closure (S on a 3000 s simulation of the fitted twin) is
             >= -0.05 in all 8 real recordings. This uses twin simulations
             only, never the real or virtual S.
  G0         micro-informed twin prediction, 64 replicas (was 16, whose
             Monte Carlo noise alone costs about 1/64 of the twin variance).

Part A, synthetic fish with a known answer. For N = 8 and 39, the full model
fitted to recording 0 (coefficients, residual pool, each fish's median
speed held fixed) is simulated for 60 min with interaction ("school") and
without ("loners"); the pipeline is then run on these as if they were data.

Predictions, fixed before running (all at the selected macro state):

  A1  Loners: |S| <= 0.05 for both N.
  A2  School: S from the fitted twin within 0.05 of S from the true twin.
  A3  School: S >= S(loners) + 0.2 for both N.
  B1  Real virtual groups: S <= 0.1 in every recording.
  B2  Real schools: S >= S(virtual) + 0.2 in every recording.
  B3  A macro state from the list passes the closure rule.
  B4  Replication: alignment > 0 and > 3x the virtual group's, every recording.
  B5  Twin noise: rho1 > 0 in every recording, and the fitted model's
      residuals have |lag-0.1 s autocorrelation| < 0.15 in every recording.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import (ARENA, DT, MIN_SPEED, TAU, TRAIN, _wrap,
                                       kinematics, polarization, social_terms,
                                       virtual, wall_terms)
from experiments.fish_data import RECORDINGS, load

REPLICAS = 64
MAX_TEST_STATES = 1500
TWIN_MACRO_STEPS = 3000
SYNTH_STEPS = 36000
CANDIDATES = ("M1", "M2", "M3", "M4")
N_OWN = 5
CLOSURE_MIN = -0.05


def turns(theta, speed):
    """Turn into each sample, lagged turns (0 where slow), and validity of the turn out."""
    d = np.zeros_like(theta)
    d[1:] = _wrap(theta[1:] - theta[:-1])
    moving = speed >= MIN_SPEED
    ok_in = np.zeros_like(moving)
    ok_in[1:] = moving[1:] & moving[:-1]
    d = np.where(ok_in, d, 0.0)
    lag1 = d
    lag2 = np.zeros_like(d)
    lag2[1:] = d[:-1]
    return lag1, lag2, ok_in


def fit_model(p, theta, speed, stop):
    lag1, lag2, ok_in = turns(theta, speed)
    own = np.concatenate([np.ones(theta.shape + (1,)), wall_terms(p, theta),
                          lag1[..., None], lag2[..., None]], -1)
    x = np.concatenate([own, social_terms(p, theta)], -1)[:-1]
    y = lag1[1:]
    ok = ok_in[1:].copy()
    ok[stop:] = False
    coef = np.linalg.lstsq(x[ok], y[ok], rcond=None)[0]
    res_all = np.where(ok, y - x @ coef, np.nan)
    pair = ok[1:] & ok[:-1]
    ac = np.corrcoef(res_all[:-1][pair], res_all[1:][pair])[0, 1]
    return coef, res_all[ok], float(ac)


def own_turn(p, theta, l1, l2, c):
    return c[0] + wall_terms(p, theta) @ c[1:3] + c[3] * l1 + c[4] * l2


def advance(p, theta, speed, turn):
    theta = _wrap(theta + turn)
    p = np.clip(p + (speed * DT)[..., None] * np.stack([np.cos(theta), np.sin(theta)], -1),
                0.0, ARENA)
    return p, theta


def twin_step(state, c, res, rng):
    p, th, sp, l1, l2 = state
    turn = own_turn(p, th, l1, l2, c) + rng.choice(res, th.shape)
    p, th = advance(p, th, sp, turn)
    return p, th, sp, turn, l1


def twin_predict(state, c, res, rng):
    rep = tuple(np.repeat(a[:, None], REPLICAS, 1) for a in state)
    for _ in range(TAU):
        rep = twin_step(rep, c, res, rng)
    return polarization(rep[1]).mean(-1)


def macro_state(p, theta, l1):
    d = np.stack([p[..., 0], ARENA - p[..., 0], p[..., 1], ARENA - p[..., 1]], -1)
    normals = np.array([0.0, np.pi, np.pi / 2, -np.pi / 2])
    j = d.argmin(-1)
    e = np.exp(-np.take_along_axis(d, j[..., None], -1)[..., 0].clip(min=0.0) / 15.0)
    return np.stack([polarization(theta), e.mean(-1), (np.cos(theta - normals[j]) * e).mean(-1),
                     np.abs(l1).mean(-1)], -1)


def poly3(x):
    cols = [np.ones(len(x))]
    for deg in (1, 2, 3):
        for idx in combinations_with_replacement(range(x.shape[1]), deg):
            cols.append(np.prod(x[:, idx], axis=1))
    return np.stack(cols, 1)


def shares(states, c, res, rng) -> dict:
    """S for every candidate macro state; states are full micro states at 1 s intervals."""
    p, th, sp, l1, l2 = states
    macro = macro_state(p, th, l1)
    x, z = macro[:-1], macro[1:, 0]
    cut = int(TRAIN * len(x))
    test = np.arange(cut, len(x))
    if len(test) > MAX_TEST_STATES:
        test = test[np.linspace(0, len(test) - 1, MAX_TEST_STATES).astype(int)]
    g0 = twin_predict(tuple(a[test] for a in states), c, res, rng)
    mse0 = np.mean((z[test] - g0) ** 2)
    out = {}
    for k, name in enumerate(CANDIDATES):
        xs = x[:, :k + 1]
        mu, sd = xs[:cut].mean(0), xs[:cut].std(0) + 1e-12
        f = poly3((xs - mu) / sd)
        beta = np.linalg.lstsq(f[:cut], z[:cut], rcond=None)[0]
        out[name] = float(1.0 - np.mean((z[test] - f[test] @ beta) ** 2) / mse0)
    return out


def analyse(p, theta, speed, rng, true_twin=None, closure=True):
    stop = int(TRAIN * len(theta))
    coef, res, ac = fit_model(p, theta, speed, stop)
    l1, l2, _ = turns(theta, speed)
    m = np.arange(0, len(theta), TAU)
    states = (p[m], theta[m], speed[m], l1[m], l2[m])
    out = {"coef": coef.tolist(), "res_autocorr": ac, "S": shares(states, coef[:N_OWN], res, rng)}
    if true_twin is not None:
        out["S_true_twin"] = shares(states, *true_twin, rng)
    if closure:
        s0 = tuple(a[m[int(TRAIN * len(m))]] for a in (p, theta, speed, l1, l2))
        out["closure"] = simulate_closure(s0, coef[:N_OWN], res, rng)
    return out


def simulate_closure(s0, c, res, rng):
    steps = TWIN_MACRO_STEPS * TAU
    state = tuple(np.array(a, float) for a in s0)
    keep = [[] for _ in range(5)]
    for t in range(steps):
        if t % TAU == 0:
            for lst, a in zip(keep, state):
                lst.append(a)
        state = twin_step(state, c, res, rng)
    return shares(tuple(np.array(k) for k in keep), c, res, rng)


def simulate_synthetic(coef, res, speed, p0, th0, social, rng, steps=SYNTH_STEPS):
    """Full fitted model (social=True) or its twin, run as a synthetic 'recording'."""
    n = len(speed)
    p, th = p0.copy(), th0.copy()
    l1, l2 = np.zeros(n), np.zeros(n)
    ps, ths = np.empty((steps, n, 2)), np.empty((steps, n))
    for t in range(steps):
        ps[t], ths[t] = p, th
        turn = own_turn(p, th, l1, l2, coef) + rng.choice(res, n)
        if social:
            turn = turn + social_terms(p[None], th[None])[0] @ coef[N_OWN:]
        p, th = advance(p, th, speed, turn)
        l1, l2 = turn, l1
    return ps, ths


def _real(args):
    n, rec = args
    rng = np.random.default_rng(32_000 + 10 * n + rec)
    pos = load(n, rec)
    out = {"n": n, "rec": rec}
    for kind, data in (("real", pos), ("virtual", virtual(pos))):
        p, th, sp = kinematics(data)
        sp = np.where(sp > 100.0, 0.0, sp)
        out[kind] = analyse(p, th, sp, rng, closure=(kind == "real"))
    return out


def _synthetic(n):
    rng = np.random.default_rng(32_500 + n)
    p, th, sp = kinematics(load(n, 0))
    sp = np.where(sp > 100.0, 0.0, sp)
    coef, res, _ = fit_model(p, th, sp, int(TRAIN * len(th)))
    speed = np.median(sp, axis=0)
    out = {"n": n}
    for kind, social in (("school", True), ("loners", False)):
        ps, ths = simulate_synthetic(coef, res, speed, p[0], th[0], social, rng)
        spd = np.broadcast_to(speed, ths.shape).copy()
        out[kind] = analyse(ps, ths, spd, rng, true_twin=(coef[:N_OWN], res), closure=False)
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    global TWIN_MACRO_STEPS, MAX_TEST_STATES, SYNTH_STEPS
    out_dir = out_dir or Path("results")
    if quick:
        TWIN_MACRO_STEPS, MAX_TEST_STATES, SYNTH_STEPS = 600, 300, 6000
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    tasks = [(n, r) for n, recs in RECORDINGS.items() for r in recs]
    print(f"  {len(tasks)} recordings + 2 synthetic groups ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(10, (os.cpu_count() or 2) - 2))) as pool:
        synth = pool.map(_synthetic, (8, 39))
        real = pool.map(_real, tasks)
        rows = {"real": list(real), "synthetic": list(synth)}
    (out_dir / "e32_fish_closed_twin_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    real, synth = rows["real"], rows["synthetic"]
    chosen = next((m for m in CANDIDATES
                   if all(r["real"]["closure"][m] >= CLOSURE_MIN for r in real)), None)
    m = chosen or CANDIDATES[-1]
    lines = ["Experiment 32: fish schools with a persistent twin and a closed macro state", "",
             "  Twin closure (S on twin simulations) per candidate macro state:",
             "  " + f"{'N':>3}{'rec':>4}" + "".join(f"{c:>8}" for c in CANDIDATES)]
    for r in real:
        lines.append(f"  {r['n']:3d}{r['rec']:4d}"
                     + "".join(f"{r['real']['closure'][c]:8.3f}" for c in CANDIDATES))
    lines += ["", f"  Selected macro state: {chosen or 'none passes; reporting ' + m}", "",
              "  Part A, synthetic fish (known answer):",
              f"  {'N':>3}{'S loners':>10}{'S school':>10}{'S school, true twin':>21}"]
    for s in synth:
        lines.append(f"  {s['n']:3d}{s['loners']['S'][m]:10.3f}{s['school']['S'][m]:10.3f}"
                     f"{s['school']['S_true_twin'][m]:21.3f}")
    lines += ["", "  Part B, real recordings:",
              f"  {'N':>3}{'rec':>4}{'S real':>8}{'S virtual':>11}{'rho1':>7}{'res ac':>8}"
              f"{'align real':>12}{'align virt.':>12}"]
    for r in real:
        re, vi = r["real"], r["virtual"]
        lines.append(f"  {r['n']:3d}{r['rec']:4d}{re['S'][m]:8.3f}{vi['S'][m]:11.3f}"
                     f"{re['coef'][3]:7.3f}{re['res_autocorr']:8.3f}"
                     f"{re['coef'][5]:12.4f}{vi['coef'][5]:12.4f}")
    a1 = all(abs(s["loners"]["S"][m]) <= 0.05 for s in synth)
    a2 = all(abs(s["school"]["S"][m] - s["school"]["S_true_twin"][m]) <= 0.05 for s in synth)
    a3 = all(s["school"]["S"][m] >= s["loners"]["S"][m] + 0.2 for s in synth)
    b1 = all(r["virtual"]["S"][m] <= 0.1 for r in real)
    b2 = all(r["real"]["S"][m] >= r["virtual"]["S"][m] + 0.2 for r in real)
    b3 = chosen is not None
    b4 = all(r["real"]["coef"][5] > 0 and r["real"]["coef"][5] > 3 * abs(r["virtual"]["coef"][5])
             for r in real)
    b5 = all(r["real"]["coef"][3] > 0 and abs(r["real"]["res_autocorr"]) < 0.15 for r in real)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(a1)}  A1 synthetic loners |S| <= 0.05",
              f"  {tag(a2)}  A2 synthetic school: fitted twin within 0.05 of true twin",
              f"  {tag(a3)}  A3 synthetic school S >= loners + 0.2",
              f"  {tag(b1)}  B1 virtual groups S <= 0.1 [max {max(r['virtual']['S'][m] for r in real):.3f}]",
              f"  {tag(b2)}  B2 real S >= virtual + 0.2 [min gap "
              f"{min(r['real']['S'][m] - r['virtual']['S'][m] for r in real):.3f}]",
              f"  {tag(b3)}  B3 a macro state passes the closure rule",
              f"  {tag(b4)}  B4 alignment > 0 and > 3x virtual",
              f"  {tag(b5)}  B5 rho1 > 0 and |residual autocorrelation| < 0.15 "
              f"[max {max(abs(r['real']['res_autocorr']) for r in real):.3f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
