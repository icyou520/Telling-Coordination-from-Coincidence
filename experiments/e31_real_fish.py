"""Experiment 31: interaction share of real fish schools.

Data: Puy et al. 2024 (PNAS), 2 recordings of 39 fish (60 min) and 6 of 8
fish (28-60 min), every fish tracked in every frame, 1 m square tank. Only
the file format (shape, units, missing values, arena bounds) was inspected
before these predictions were written.

Pipeline (all choices fixed in advance):

  Sampling   positions every 0.1 s; velocity from consecutive samples;
             heading theta = angle of velocity; heading changes are fitted
             only where speed >= 1 cm/s before and after.
  Macro      polarization phi = |mean unit heading| every 1 s (tau = 10).
  Micro fit  least squares on the 0.1 s heading change, first 70 % of time:
               own terms         intercept (turning bias), wall turning
                                 sin(normal - theta) exp(-d / l), l = 5, 15 cm
                                 (nearest wall: d distance, normal inward)
               interaction terms alignment wrap(psi - theta) with psi the
                                 mean heading of the 4 nearest fish;
                                 attraction sin(angle to their centroid -
                                 theta); repulsion sin(angle to the nearest
                                 fish - theta) exp(-d_nn / 2 cm)
  Twin       own terms plus residual noise (resampled from the fit's
             residuals); every fish keeps its current speed; positions
             clipped to the tank. Interaction terms vanish when a fish has no
             neighbours, so this is the isolation twin of e27.
  S          G cubic in phi fitted on the first 70 %; G0 the micro-informed
             twin prediction (mean phi after 1 s over 16 twin replicas,
             started from the actual positions, headings and speeds);
             scored on the last 30 %.
  Closure    the same pipeline applied to a long simulation of the twin
             itself (e30 lesson: S is only meaningful if the twin is closed
             on phi).
  Control    virtual groups: each fish's trajectory circularly shifted in
             time by i * T / N (>= 90 s apart), so fish share the tank but
             never interacted; the whole pipeline is rerun on them.

Predictions, fixed before running:

  R1  Real schools: S >= 0.5 in both 39-fish recordings and in >= 4 of 6
      8-fish recordings.
  R2  Virtual groups: S <= 0.1 in every recording.
  R3  Alignment coefficient > 0 in every real recording and more than 3 times
      the virtual group's (in absolute value).
  R4  Twin closure: S on the simulated twin >= -0.05 in every recording.
Exploratory: a naive twin (own terms only, fitted without interaction terms).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.fish_data import FPS, RECORDINGS, load

DT = 0.1
STRIDE = int(FPS * DT)
TAU = 10
K_NN = 4
ARENA = 100.0
MIN_SPEED = 1.0
REPLICAS = 16
TRAIN = 0.7
MAX_TEST_STATES = 1500
TWIN_MACRO_STEPS = 3000


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def kinematics(pos: np.ndarray):
    """Positions at 0.1 s -> (P, theta, speed) aligned so velocity leads into P."""
    p = pos[::STRIDE]
    v = np.diff(p, axis=0) / DT
    return p[1:], np.arctan2(v[..., 1], v[..., 0]), np.hypot(v[..., 0], v[..., 1])


def wall_terms(p: np.ndarray, theta: np.ndarray) -> np.ndarray:
    d = np.stack([p[..., 0], ARENA - p[..., 0], p[..., 1], ARENA - p[..., 1]], -1)
    normals = np.array([0.0, np.pi, np.pi / 2, -np.pi / 2])
    j = d.argmin(-1)
    dmin = np.take_along_axis(d, j[..., None], -1)[..., 0].clip(min=0.0)
    s = np.sin(normals[j] - theta)
    return np.stack([s * np.exp(-dmin / 5.0), s * np.exp(-dmin / 15.0)], -1)


def social_terms(p: np.ndarray, theta: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Alignment, attraction, repulsion with the K_NN nearest fish; shape (T, N, 3)."""
    t_len, n = theta.shape
    out = np.empty((t_len, n, 3))
    k = min(K_NN, n - 1)
    for a in range(0, t_len, chunk):
        pp, th = p[a:a + chunk], theta[a:a + chunk]
        diff = pp[:, None, :, :] - pp[:, :, None, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        idx = np.arange(n)
        dist[:, idx, idx] = np.inf
        nn = np.argpartition(dist, k - 1, axis=-1)[..., :k]
        rows = np.arange(len(pp))[:, None, None]
        u = np.exp(1j * th)
        psi = np.angle(u[rows, nn].sum(-1))
        cen = pp[rows, nn].mean(-2) - pp
        near = np.take_along_axis(dist, dist.argmin(-1)[..., None], -1)[..., 0]
        jn = dist.argmin(-1)
        to_near = pp[rows[..., 0], jn] - pp
        out[a:a + chunk, :, 0] = _wrap(psi - th)
        out[a:a + chunk, :, 1] = np.sin(np.arctan2(cen[..., 1], cen[..., 0]) - th)
        out[a:a + chunk, :, 2] = np.sin(np.arctan2(to_near[..., 1], to_near[..., 0]) - th) \
            * np.exp(-near / 2.0)
    return out


def fit_model(p, theta, speed, stop: int, social: bool = True):
    """Least squares on heading change for t < stop. Returns (coef, residuals)."""
    own = np.concatenate([np.ones(theta.shape + (1,)), wall_terms(p, theta)], -1)
    x = np.concatenate([own, social_terms(p, theta)], -1) if social else own
    y = _wrap(theta[1:] - theta[:-1])
    ok = (speed[:-1] >= MIN_SPEED) & (speed[1:] >= MIN_SPEED)
    ok[stop:] = False
    xs, ys = x[:-1][ok], y[ok]
    coef = np.linalg.lstsq(xs, ys, rcond=None)[0]
    return coef, ys - xs @ coef


def twin_step(p, theta, speed, coef_own, residuals, rng):
    w = wall_terms(p, theta)
    drift = coef_own[0] + w @ coef_own[1:3]
    theta = _wrap(theta + drift + rng.choice(residuals, theta.shape))
    p = np.clip(p + (speed * DT)[..., None] * np.stack([np.cos(theta), np.sin(theta)], -1),
                0.0, ARENA)
    return p, theta


def twin_predict(p, theta, speed, coef_own, residuals, rng) -> np.ndarray:
    """Micro-informed twin: mean phi after TAU steps, from each state; shapes (S, N, ...)."""
    pp = np.repeat(p[:, None], REPLICAS, 1)
    th = np.repeat(theta[:, None], REPLICAS, 1)
    sp = np.repeat(speed[:, None], REPLICAS, 1)
    for _ in range(TAU):
        pp, th = twin_step(pp, th, sp, coef_own, residuals, rng)
    return np.abs(np.exp(1j * th).mean(-1)).mean(-1)


def polarization(theta):
    return np.abs(np.exp(1j * theta).mean(-1))


def share(phi_macro, states, coef_own, residuals, rng) -> float:
    """phi_macro[m] at macro times m; states[m] = (p, theta, speed) at the same times."""
    x, z = phi_macro[:-1], phi_macro[1:]
    cut = int(TRAIN * len(x))
    coef = np.polyfit(x[:cut], z[:cut], 3)
    test = np.arange(cut, len(x))
    if len(test) > MAX_TEST_STATES:
        test = test[np.linspace(0, len(test) - 1, MAX_TEST_STATES).astype(int)]
    p, th, sp = states
    g0 = twin_predict(p[test], th[test], sp[test], coef_own, residuals, rng)
    return float(1.0 - np.mean((z[test] - np.polyval(coef, x[test])) ** 2)
                 / np.mean((z[test] - g0) ** 2))


def analyse_series(p, theta, speed, rng, social=True):
    stop = int(TRAIN * len(theta))
    coef, res = fit_model(p, theta, speed, stop, social=social)
    macro = np.arange(0, len(theta), TAU)
    states = (p[macro], theta[macro], speed[macro])
    s = share(polarization(theta[macro]), states, coef[:3], res, rng)
    return coef, res, s


def closure(p0, th0, sp0, coef, res, rng) -> float:
    """Run the twin itself for TWIN_MACRO_STEPS and apply the same S pipeline."""
    p, th = p0.copy(), th0.copy()
    steps = TWIN_MACRO_STEPS * TAU
    ps, ths = np.empty((steps,) + p.shape), np.empty((steps,) + th.shape)
    for t in range(steps):
        p, th = twin_step(p, th, sp0, coef[:3], res, rng)
        ps[t], ths[t] = p, th
    macro = np.arange(0, steps, TAU)
    sp = np.broadcast_to(sp0, ths.shape)
    return share(polarization(ths[macro]), (ps[macro], ths[macro], sp[macro]), coef[:3], res, rng)


def virtual(pos: np.ndarray) -> np.ndarray:
    n = pos.shape[1]
    out = np.empty_like(pos)
    for i in range(n):
        out[:, i] = np.roll(pos[:, i], i * len(pos) // n, axis=0)
    return out


def _run(args):
    n, rec = args
    rng = np.random.default_rng(31_000 + 10 * n + rec)
    pos = load(n, rec)
    out = {"n": n, "rec": rec}
    for kind, data in (("real", pos), ("virtual", virtual(pos))):
        p, th, sp = kinematics(data)
        jump = sp > 100.0
        sp = np.where(jump, 0.0, sp)
        coef, res, s = analyse_series(p, th, sp, rng)
        out[kind] = {"S": s, "coef": coef.tolist(), "phi_mean": float(polarization(th).mean()),
                     "speed_median": float(np.median(sp))}
        if kind == "real":
            _, _, s_naive = analyse_series(p, th, sp, rng, social=False)
            out[kind]["S_naive"] = s_naive
            start = int(TRAIN * len(th))
            out[kind]["S_twin_closure"] = closure(p[start], th[start], sp[start], coef, res, rng)
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    global TWIN_MACRO_STEPS, MAX_TEST_STATES
    out_dir = out_dir or Path("results")
    if quick:
        TWIN_MACRO_STEPS, MAX_TEST_STATES = 600, 300
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    for n, recs in RECORDINGS.items():
        for r in recs:
            load(n, r)
    tasks = [(n, r) for n, recs in RECORDINGS.items() for r in recs]
    print(f"  analysing {len(tasks)} recordings ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(len(tasks), (os.cpu_count() or 2) - 2))) as pool:
        rows = list(pool.map(_run, tasks))
    (out_dir / "e31_real_fish_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    lines = ["Experiment 31: interaction share of real fish schools (Puy et al. 2024)", "",
             f"  {'N':>3}{'rec':>4}{'<phi>':>7}{'speed':>7}{'S real':>8}{'S naive':>9}"
             f"{'S virtual':>11}{'twin closure':>14}{'align real':>12}{'align virt.':>12}"]
    for r in rows:
        re, vi = r["real"], r["virtual"]
        lines.append(f"  {r['n']:3d}{r['rec']:4d}{re['phi_mean']:7.3f}{re['speed_median']:7.1f}"
                     f"{re['S']:8.3f}{re['S_naive']:9.3f}{vi['S']:11.3f}{re['S_twin_closure']:14.3f}"
                     f"{re['coef'][3]:12.4f}{vi['coef'][3]:12.4f}")
    big = [r for r in rows if r["n"] == 39]
    small = [r for r in rows if r["n"] == 8]
    r1 = all(r["real"]["S"] >= 0.5 for r in big) and sum(r["real"]["S"] >= 0.5 for r in small) >= 4
    r2 = all(r["virtual"]["S"] <= 0.1 for r in rows)
    r3 = all(r["real"]["coef"][3] > 0 and r["real"]["coef"][3] > 3 * abs(r["virtual"]["coef"][3])
             for r in rows)
    r4 = all(r["real"]["S_twin_closure"] >= -0.05 for r in rows)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(r1)}  R1 real schools S >= 0.5 (both N=39; >= 4/6 N=8) "
              f"[N=8 count {sum(r['real']['S'] >= 0.5 for r in small)}/6]",
              f"  {tag(r2)}  R2 virtual groups S <= 0.1 [max {max(r['virtual']['S'] for r in rows):.3f}]",
              f"  {tag(r3)}  R3 alignment > 0 and > 3x virtual in every recording",
              f"  {tag(r4)}  R4 twin closure S >= -0.05 "
              f"[min {min(r['real']['S_twin_closure'] for r in rows):.3f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
