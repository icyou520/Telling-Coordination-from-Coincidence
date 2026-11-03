"""Experiment 37: pigeon flocks with speed dynamics.

e36 (1 s micro step) removed the smoothing artefact and found clear
alignment, but the fitted turning model had no cohesion (attraction about 0;
simulated flocks spread to about 260 m against 6-8 m real), so its gate
failed. Likely cause: pigeons keep together by changing speed and by
front-back positioning, which a constant-speed turning model cannot express.

Changes from e36 (same flocks, flights, 1 s micro step, 2 s macro step,
virtual flocks, S_proj with 64 replicas, 70/30 split, gate):

  Turn model   own: intercept, homing sin(beta - theta) * (1, r/50, (r/50)^2),
               last turn. Interaction (4 nearest birds): alignment;
               attraction sin(angle to their centroid - theta), and the same
               times d_centroid / 10 m (attraction growing with distance);
               repulsion sin(angle to nearest - theta) exp(-d_nn / 2 m).
  Speed model  change of speed over 1 s. Own: intercept, speed (relaxation to
               a preferred speed), last speed change. Interaction:
               front-back offset of the neighbours' centroid along the
               bird's heading / 10 m, and the neighbours' mean speed minus
               its own.
  Twin         own terms of both models; turn and speed residuals resampled
               as pairs; speed clipped to [5, 25] m/s.

Predictions, fixed before running (each for flocks A, B and C):

  G   Gate: the synthetic flock simulated from the full fitted model has
      mean distance to centroid <= 3x the real flock's and S_proj >= 0.2.
      R3-R5 count only for flocks that pass.
  R1  Synthetic loners: |S_proj| <= 0.03.
  R2  Synthetic flock: fitted twin within 0.05 of the true twin.
  R3  Virtual flocks: S_proj <= 0.1.
  R4  Real flocks: S_proj >= S_proj(virtual) + 0.2.
  R5  State dependence <= 0.05.
  R6  |Residual lag-1 autocorrelation| of turns <= 0.2.
  R7  Front-back coefficient > 0: birds speed up when neighbours are ahead
      (as reported for pigeon pairs by Pettit et al. 2013).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import _wrap, polarization
from experiments.e35_pigeons import homing, virtual
from experiments.pigeon_data import FLOCKS, flights, load

DT, TAU, STRIDE = 1.0, 2, 10
K_NN = 4
REPLICAS = 64
MAX_STATES = 1500
TRAIN = 0.7
V_MIN, V_MAX = 5.0, 25.0
N_TURN_OWN, N_SPEED_OWN = 5, 3
TURN_SOCIAL, SPEED_SOCIAL = slice(0, 4), slice(4, 6)


def _gather(a, nn):
    return np.take_along_axis(np.broadcast_to(a[..., None, :], nn.shape[:-1] + (a.shape[-1],)), nn, -1)


def social(p, th, v):
    """Interaction features (..., N, 6): align, attract, attract * d/10, repel, front-back, speed diff."""
    dx = p[..., None, :, 0] - p[..., :, None, 0]
    dy = p[..., None, :, 1] - p[..., :, None, 1]
    dist = np.hypot(dx, dy)
    n = th.shape[-1]
    dist[..., np.arange(n), np.arange(n)] = np.inf
    nn = np.argpartition(dist, K_NN - 1, -1)[..., :K_NN]
    psi = np.angle(np.exp(1j * _gather(th, nn)).sum(-1))
    cx = np.take_along_axis(dx, nn, -1).mean(-1)
    cy = np.take_along_axis(dy, nn, -1).mean(-1)
    dc = np.hypot(cx, cy)
    jn = dist.argmin(-1)[..., None]
    near = np.take_along_axis(dist, jn, -1)[..., 0]
    nx, ny = np.take_along_axis(dx, jn, -1)[..., 0], np.take_along_axis(dy, jn, -1)[..., 0]
    att = np.sin(np.arctan2(cy, cx) - th)
    return np.stack([_wrap(psi - th), att, att * dc / 10.0,
                     np.sin(np.arctan2(ny, nx) - th) * np.exp(-near / 2.0),
                     (cx * np.cos(th) + cy * np.sin(th)) / 10.0,
                     _gather(v, nn).mean(-1) - v], -1)


def own_features(p, th, v, l1, dvl):
    turn = np.concatenate([np.ones(th.shape + (1,)), homing(p, th), l1[..., None]], -1)
    speed = np.stack([np.ones(th.shape), v, dvl], -1)
    return turn, speed


def prepare(s):
    th, v, cont = s["theta"], s["speed"], s["cont"]
    turn_in = np.full(th.shape, np.nan)
    dv_in = np.full(th.shape, np.nan)
    turn_in[1:] = np.where(cont[1:, None], _wrap(th[1:] - th[:-1]), np.nan)
    dv_in[1:] = np.where(cont[1:, None], v[1:] - v[:-1], np.nan)
    t_turn, t_dv = np.full(th.shape, np.nan), np.full(th.shape, np.nan)
    t_turn[:-1], t_dv[:-1] = turn_in[1:], dv_in[1:]
    return dict(s, l1=turn_in, dvl=dv_in, t_turn=t_turn, t_dv=t_dv)


def fit(series):
    xt, xv, yt, yv = [], [], [], []
    for s in series:
        stop = s["train_stop"]
        ft, fv = own_features(s["p"], s["theta"], s["speed"], s["l1"], s["dvl"])
        soc = social(s["p"], s["theta"], s["speed"])
        a = np.concatenate([ft, soc[..., TURN_SOCIAL]], -1)[:stop]
        b = np.concatenate([fv, soc[..., SPEED_SOCIAL]], -1)[:stop]
        ok = (np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
              & np.isfinite(s["t_turn"][:stop]) & np.isfinite(s["t_dv"][:stop]))
        xt.append(a[ok]), xv.append(b[ok]), yt.append(s["t_turn"][:stop][ok]), yv.append(s["t_dv"][:stop][ok])
    xt, xv, yt, yv = map(np.concatenate, (xt, xv, yt, yv))
    ct = np.linalg.lstsq(xt, yt, rcond=None)[0]
    cv = np.linalg.lstsq(xv, yv, rcond=None)[0]
    return ct, cv, np.stack([yt - xt @ ct, yv - xv @ cv], 1)


def step(state, ct, cv, res, rng, social_on=False):
    p, th, v, l1, dvl = state
    ft, fv = own_features(p, th, v, l1, dvl)
    turn = ft @ ct[:N_TURN_OWN]
    dv = fv @ cv[:N_SPEED_OWN]
    if social_on:
        soc = social(p, th, v)
        turn = turn + soc[..., TURN_SOCIAL] @ ct[N_TURN_OWN:]
        dv = dv + soc[..., SPEED_SOCIAL] @ cv[N_SPEED_OWN:]
    draw = res[rng.integers(0, len(res), th.shape)]
    turn, dv = turn + draw[..., 0], dv + draw[..., 1]
    v_new = np.clip(v + dv, V_MIN, V_MAX)
    th = _wrap(th + turn)
    p = p + (v_new * DT)[..., None] * np.stack([np.cos(th), np.sin(th)], -1)
    return p, th, v_new, turn, v_new - v


def twin_predict(states, ct, cv, res, rng):
    st = tuple(np.repeat(a[:, None], REPLICAS, 1) for a in states)
    for _ in range(TAU):
        st = step(st, ct, cv, res, rng)
    return polarization(st[1]).mean(-1)


def macro_pairs(s):
    ok = np.isfinite(s["l1"]).all(1) & np.isfinite(s["dvl"]).all(1)
    return np.array([t for t in range(len(s["theta"]) - TAU)
                     if ok[t] and s["cont"][t + 1:t + TAU + 1].all()], int)


def _sub(idx):
    return idx if len(idx) <= MAX_STATES else idx[np.linspace(0, len(idx) - 1, MAX_STATES).astype(int)]


def analyse(raw, rng, true_twin=None):
    series = [prepare(s) for s in raw]
    pairs = [macro_pairs(s) for s in series]
    cut = int(TRAIN * sum(len(pp) for pp in pairs))
    seen = 0
    for s, pp in zip(series, pairs):
        k = int(np.clip(cut - seen, 0, len(pp)))
        s["train_stop"] = int(pp[k - 1]) + TAU + 1 if k > 0 else 0
        s["is_train"] = np.arange(len(pp)) < k
        seen += len(pp)
    ct, cv, res = fit(series)
    keys = ("p", "theta", "speed", "l1", "dvl")
    stacked = {k: np.concatenate([s[k][pp] for s, pp in zip(series, pairs)]) for k in keys}
    x = polarization(stacked["theta"])
    z = np.concatenate([polarization(s["theta"][pp + TAU]) for s, pp in zip(series, pairs)])
    is_train = np.concatenate([s["is_train"] for s in series])
    tr, te = _sub(np.flatnonzero(is_train)), _sub(np.flatnonzero(~is_train))
    g = np.polyval(np.polyfit(x[is_train], z[is_train], 3), x[te])
    mse = lambda pred: float(np.mean((z[te] - pred) ** 2))

    def shares(ct_, cv_, res_):
        g0 = lambda idx: twin_predict(tuple(stacked[k][idx] for k in keys), ct_, cv_, res_, rng)
        g0_tr, g0_te = g0(tr), g0(te)
        return {"S_proj": 1 - mse(g) / mse(np.polyval(np.polyfit(x[tr], g0_tr, 3), x[te])),
                "S_proj_alt": 1 - mse(g) / mse(np.polyval(np.polyfit(x[te], g0_te, 3), x[te])),
                "S_micro": 1 - mse(g) / mse(g0_te)}

    out = {"ct": ct.tolist(), "cv": cv.tolist(), "phi_mean": float(x.mean()), **shares(ct, cv, res)}
    if true_twin is not None:
        out["true_twin"] = shares(*true_twin)
    return out


def spread(p):
    c = p - p.mean(1, keepdims=True)
    return float(np.hypot(c[..., 0], c[..., 1]).mean())


def turn_residual_autocorr(series, ct):
    a_all, b_all = [], []
    for s in series:
        ft, _ = own_features(s["p"], s["theta"], s["speed"], s["l1"], s["dvl"])
        x = np.concatenate([ft, social(s["p"], s["theta"], s["speed"])[..., TURN_SOCIAL]], -1)
        r = s["t_turn"] - x @ ct
        ok = np.isfinite(r[:-1]) & np.isfinite(r[1:])
        a_all.append(r[:-1][ok]), b_all.append(r[1:][ok])
    return float(np.corrcoef(np.concatenate(a_all), np.concatenate(b_all))[0, 1])


def simulate(ct, cv, res, s0, steps, social_on, rng):
    p, th, v = s0["p"][0].copy(), s0["theta"][0].copy(), s0["speed"][0].copy()
    state = (p, th, v, np.zeros_like(th), np.zeros_like(th))
    ps, ths, vs = np.empty((steps,) + p.shape), np.empty((steps,) + th.shape), np.empty((steps,) + th.shape)
    for t in range(steps):
        ps[t], ths[t], vs[t] = state[0], state[1], state[2]
        state = step(state, ct, cv, res, rng, social_on)
    return {"p": ps, "theta": ths, "speed": vs, "cont": np.r_[False, np.ones(steps - 1, bool)]}


def _flock(flock):
    rng = np.random.default_rng(37_000 + ord(flock))
    real = [load(f, STRIDE) for f in flights(flock)]
    out = {"flock": flock, "minutes": sum(len(s["cont"]) for s in real) * DT / 60,
           "spread_real": float(np.mean([spread(s["p"]) for s in real]))}
    out["real"] = analyse(real, rng)
    out["virtual"] = analyse([virtual(s) for s in real], rng)
    series = [prepare(s) for s in real]
    for s in series:
        s["train_stop"] = len(s["theta"])
    ct, cv, res = fit(series)
    out["ct_all"], out["cv_all"] = ct.tolist(), cv.tolist()
    out["res_autocorr"] = turn_residual_autocorr(series, ct)
    steps = sum(len(s["cont"]) for s in real)
    twin = (ct[:N_TURN_OWN], cv[:N_SPEED_OWN], res)
    for kind, social_on in (("flock", True), ("loners", False)):
        sim = simulate(ct, cv, res, real[0], steps, social_on, rng)
        out["synthetic_" + kind] = analyse([sim], rng, true_twin=twin) | {"spread": spread(sim["p"])}
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    for fl in FLOCKS:
        for f in flights(fl):
            load(f, STRIDE)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(_flock, FLOCKS))
    (out_dir / "e37_pigeons_speed_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    lines = ["Experiment 37: pigeon flocks with speed dynamics", "",
             f"  {'flock':>5}{'spread real/synth':>19}{'synth S':>9}{'gate':>6}{'S_proj':>8}"
             f"{'virtual':>9}{'alt':>7}{'loners':>8}{'true':>7}{'res ac':>8}"]
    gate = {}
    for r in rows:
        sf = r["synthetic_flock"]
        gate[r["flock"]] = sf["spread"] <= 3 * r["spread_real"] and sf["S_proj"] >= 0.2
        lines.append(f"  {r['flock']:>5}{r['spread_real']:9.1f}/{sf['spread']:<9.1f}{sf['S_proj']:9.3f}"
                     f"{'pass' if gate[r['flock']] else 'FAIL':>6}{r['real']['S_proj']:8.3f}"
                     f"{r['virtual']['S_proj']:9.3f}{r['real']['S_proj_alt']:7.3f}"
                     f"{r['synthetic_loners']['S_proj']:8.3f}{sf['true_twin']['S_proj']:7.3f}"
                     f"{r['res_autocorr']:8.3f}")
        ct, cv = r["ct_all"], r["cv_all"]
        lines.append(f"        turn: lag {ct[4]:.3f}, align {ct[5]:.3f}, attract {ct[6]:.3f}, "
                     f"attract*d {ct[7]:.3f}, repel {ct[8]:.3f} | speed: relax {cv[1]:.3f}, "
                     f"lag {cv[2]:.3f}, front-back {cv[3]:.3f}, speed match {cv[4]:.3f}")
    ok = [r for r in rows if gate[r["flock"]]]
    tag = lambda v: "PASS" if v else "FAIL"
    counted = lambda f: tag(all(f(r) for r in ok)) if ok else "n/a "
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(all(gate.values()))}  G  gate [passed: {', '.join(k for k, v in gate.items() if v) or 'none'}]",
              f"  {tag(all(abs(r['synthetic_loners']['S_proj']) <= 0.03 for r in rows))}  R1 synthetic loners |S_proj| <= 0.03",
              f"  {tag(all(abs(r['synthetic_flock']['S_proj'] - r['synthetic_flock']['true_twin']['S_proj']) <= 0.05 for r in rows))}  "
              f"R2 synthetic flock: fitted twin within 0.05 of true twin",
              f"  {counted(lambda r: r['virtual']['S_proj'] <= 0.1)}  R3 virtual flocks S_proj <= 0.1 (gated)",
              f"  {counted(lambda r: r['real']['S_proj'] >= r['virtual']['S_proj'] + 0.2)}  R4 real >= virtual + 0.2 (gated)",
              f"  {counted(lambda r: abs(r['real']['S_proj'] - r['real']['S_proj_alt']) <= 0.05)}  R5 state dependence <= 0.05 (gated)",
              f"  {tag(all(abs(r['res_autocorr']) <= 0.2 for r in rows))}  R6 |turn residual autocorrelation| <= 0.2",
              f"  {tag(all(r['cv_all'][3] > 0 for r in rows))}  R7 front-back speed coefficient > 0"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
