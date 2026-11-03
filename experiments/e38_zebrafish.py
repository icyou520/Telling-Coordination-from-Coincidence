"""Experiment 38: the projected interaction share on large zebrafish groups.

Third real data set, chosen because turning-based models are known to
describe it (Heras et al. 2019, PLoS Comput. Biol. 15, e1007354; tracked with
idtracker.ai, Romero-Ferrero et al. 2019): nine 10-minute recordings of 60,
80 and 100 juvenile zebrafish (three each) in a circular arena, 32 frames
per second. Only the file format, missing-value rate (0.01-0.35 %), arena
extent and body length were inspected before these predictions were written.

Pipeline (the fish pipeline of e32/e33 with a circular arena; all choices
fixed in advance):

  Units     body lengths (from each file); arena centre = middle of the
            position extent, radius = largest distance from it.
            Missing positions are linearly interpolated.
  Sampling  every 3rd frame (about 0.094 s); headings from velocity; turns
            fitted only where speed >= 0.5 BL/s before and after.
  Macro     polarization phi, horizon 10 samples (about 0.94 s), starting
            every 2nd sample; first 70 % of time trains, the rest tests.
  Twin      own terms: bias, wall turning sin(inward normal - theta)
            exp(-d_wall / l), l = 2 and 6 BL, last two turns; plus resampled
            residuals; speed held; positions kept inside the arena.
  Social    alignment, attraction and repulsion (exp(-d_nn / 2 BL)) with the
            4 nearest fish.
  S_proj    as e33 (32 replicas, at most 1000 train and 1000 test states).
  Virtual   groups of the same size built across the three recordings of
            that size: fish i comes from recording (i + m) mod 3, m = 0, 1, 2,
            shifted in time by (i // 3) / (N / 3 + 1) of the recording
            (about 30 s apart), positions relative to their own arena
            centre. Nine virtual groups in all.
  Synthetic the full fitted model of each recording simulated with and
            without interaction for the recording's length, from its first
            state.

Predictions, fixed before running:

  G   Gate, per recording: the synthetic school has S_proj >= 0.2 and mean
      polarization >= half the real recording's. Z4 and Z5 count only for
      recordings that pass.
  Z1  Synthetic loners: |S_proj| <= 0.03 in every recording.
  Z2  Synthetic school: fitted twin within 0.05 of the true twin.
  Z3  Virtual groups: S_proj <= 0.1 in all nine.
  Z4  Real: S_proj >= (mean virtual S_proj of the same group size) + 0.2.
  Z5  State dependence <= 0.05.
  Z6  Alignment > 0 in every recording and > 3x the mean |virtual| alignment.
  Z7  |Residual lag-1 autocorrelation of turns| <= 0.2.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import _wrap, polarization, social_terms

DATA = Path(__file__).resolve().parent.parent / "data" / "zebrafish" / "trajectories"
SIZES = (60, 80, 100)
RECS = (1, 2, 3)
STRIDE, TAU, START_EVERY = 3, 10, 2
MIN_SPEED = 0.5
REPLICAS = 32
MAX_STATES = 1000
TRAIN = 0.7
N_OWN = 5


def load(n, rec):
    d = np.load(DATA / str(n) / f"{rec}.npy", allow_pickle=True).item()
    t = d["trajectories"].astype(float)
    idx = np.arange(len(t))
    for i in range(t.shape[1]):
        for k in range(2):
            col = t[:, i, k]
            bad = np.isnan(col)
            if bad.any():
                col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
    bl = float(d["body_length"])
    centre = (np.nanmin(t, (0, 1)) + np.nanmax(t, (0, 1))) / 2
    pos = (t - centre) / bl
    radius = float(np.hypot(pos[..., 0], pos[..., 1]).max())
    return pos, radius, STRIDE / d["frames_per_second"]


def kinematics(pos, dt):
    p = pos[::STRIDE]
    v = np.diff(p, axis=0) / dt
    return p[1:], np.arctan2(v[..., 1], v[..., 0]), np.hypot(v[..., 0], v[..., 1])


def wall_terms(p, theta, radius):
    r = np.hypot(p[..., 0], p[..., 1])
    d = np.clip(radius - r, 0.0, None)
    s = np.sin(np.arctan2(-p[..., 1], -p[..., 0]) - theta)
    return np.stack([s * np.exp(-d / 2.0), s * np.exp(-d / 6.0)], -1)


def turns(theta, speed):
    d = np.zeros_like(theta)
    d[1:] = _wrap(theta[1:] - theta[:-1])
    moving = speed >= MIN_SPEED
    ok_in = np.zeros_like(moving)
    ok_in[1:] = moving[1:] & moving[:-1]
    d = np.where(ok_in, d, 0.0)
    lag2 = np.zeros_like(d)
    lag2[1:] = d[:-1]
    return d, lag2, ok_in


def features(p, theta, speed, radius):
    l1, l2, ok_in = turns(theta, speed)
    own = np.concatenate([np.ones(theta.shape + (1,)), wall_terms(p, theta, radius),
                          l1[..., None], l2[..., None]], -1)
    return np.concatenate([own, social_terms(p, theta, chunk=300)], -1), l1, ok_in


def fit_model(p, theta, speed, radius, stop):
    x, l1, ok_in = features(p, theta, speed, radius)
    x, y = x[:-1], l1[1:]
    ok = ok_in[1:].copy()
    ok[stop:] = False
    coef = np.linalg.lstsq(x[ok], y[ok], rcond=None)[0]
    r = np.where(ok, y - x @ coef, np.nan)
    pair = np.isfinite(r[:-1]) & np.isfinite(r[1:])
    ac = float(np.corrcoef(r[:-1][pair], r[1:][pair])[0, 1])
    return coef, r[ok], ac


def step(state, c, res, rng, radius, social_coef=None):
    p, th, sp, l1, l2 = state
    turn = c[0] + wall_terms(p, th, radius) @ c[1:3] + c[3] * l1 + c[4] * l2 \
        + rng.choice(res, th.shape)
    if social_coef is not None:
        turn = turn + social_terms(p[None], th[None])[0] @ social_coef
    th = _wrap(th + turn)
    p = p + (sp * DT_HOLDER[0])[..., None] * np.stack([np.cos(th), np.sin(th)], -1)
    r = np.hypot(p[..., 0], p[..., 1])
    p = np.where((r > radius)[..., None], p * (radius / np.maximum(r, 1e-9))[..., None], p)
    return p, th, sp, turn, l1


DT_HOLDER = [0.094]


def twin_predict(states, c, res, rng, radius):
    st = tuple(np.repeat(a[:, None], REPLICAS, 1) for a in states)
    for _ in range(TAU):
        st = step(st, c, res, rng, radius)
    return polarization(st[1]).mean(-1)


def _sub(idx):
    return idx if len(idx) <= MAX_STATES else idx[np.linspace(0, len(idx) - 1, MAX_STATES).astype(int)]


def analyse(p, theta, speed, radius, rng, true_twin=None):
    stop = int(TRAIN * len(theta))
    coef, res, ac = fit_model(p, theta, speed, radius, stop)
    l1, l2, _ = turns(theta, speed)
    starts = np.arange(0, len(theta) - TAU, START_EVERY)
    phi = polarization(theta)
    x, z = phi[starts], phi[starts + TAU]
    train = starts + TAU < stop
    test = starts >= stop
    tr, te = _sub(np.flatnonzero(train)), _sub(np.flatnonzero(test))
    g = np.polyval(np.polyfit(x[train], z[train], 3), x[te])
    mse = lambda pred: float(np.mean((z[te] - pred) ** 2))

    def shares(c, r):
        g0 = lambda idx: twin_predict(tuple(a[starts[idx]] for a in (p, theta, speed, l1, l2)),
                                      c, r, rng, radius)
        g0_tr, g0_te = g0(tr), g0(te)
        return {"S_proj": 1 - mse(g) / mse(np.polyval(np.polyfit(x[tr], g0_tr, 3), x[te])),
                "S_proj_alt": 1 - mse(g) / mse(np.polyval(np.polyfit(x[te], g0_te, 3), x[te])),
                "S_micro": 1 - mse(g) / mse(g0_te)}

    out = {"coef": coef.tolist(), "res_autocorr": ac, "phi_mean": float(phi.mean()),
           **shares(coef[:N_OWN], res)}
    if true_twin is not None:
        out["true_twin"] = shares(*true_twin)
    return out, coef, res


def simulate(coef, res, p0, th0, speed, radius, steps, social, rng):
    n = len(speed)
    state = (p0.copy(), th0.copy(), speed, np.zeros(n), np.zeros(n))
    ps, ths = np.empty((steps, n, 2)), np.empty((steps, n))
    for t in range(steps):
        ps[t], ths[t] = state[0], state[1]
        state = step(state, coef[:N_OWN], res, rng, radius, coef[N_OWN:] if social else None)
    return ps, ths


def _clean(sp):
    return np.where(sp > 50.0, 0.0, sp)


def _real(args):
    n, rec = args
    rng = np.random.default_rng(38_000 + n + rec)
    pos, radius, dt = load(n, rec)
    DT_HOLDER[0] = dt
    p, th, sp = kinematics(pos, dt)
    sp = _clean(sp)
    out, coef, res = analyse(p, th, sp, radius, rng)
    row = {"n": n, "rec": rec, "real": out}
    speed = np.median(sp, 0)
    for kind, social in (("school", True), ("loners", False)):
        ps, ths = simulate(coef, res, p[0], th[0], speed, radius, len(th), social, rng)
        spd = np.broadcast_to(speed, ths.shape).copy()
        row["synthetic_" + kind] = analyse(ps, ths, spd, radius, rng,
                                           true_twin=(coef[:N_OWN], res))[0]
    return row


def _virtual(args):
    n, m = args
    rng = np.random.default_rng(38_500 + n + m)
    loaded = [load(n, r) for r in RECS]
    DT_HOLDER[0] = float(np.mean([l[2] for l in loaded]))
    length = min(len(l[0]) for l in loaded)
    radius = float(np.mean([l[1] for l in loaded]))
    per = n // 3 + 1
    pos = np.empty((length, n, 2))
    for i in range(n):
        src = loaded[(i + m) % 3][0][:length, i]
        pos[:, i] = np.roll(src, (i // 3) * length // per, axis=0)
    p, th, sp = kinematics(pos, DT_HOLDER[0])
    out, _, _ = analyse(p, th, _clean(sp), radius, rng)
    return {"n": n, "m": m, "virtual": out}


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    with ProcessPoolExecutor(max_workers=max(1, min(9, (os.cpu_count() or 2) - 2))) as pool:
        real = pool.map(_real, [(n, r) for n in SIZES for r in RECS])
        virt = pool.map(_virtual, [(n, m) for n in SIZES for m in range(3)])
        rows = {"real": list(real), "virtual": list(virt)}
    (out_dir / "e38_zebrafish_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    real, virt = rows["real"], rows["virtual"]
    vmean = {n: np.mean([v["virtual"]["S_proj"] for v in virt if v["n"] == n]) for n in SIZES}
    valign = np.mean([abs(v["virtual"]["coef"][N_OWN]) for v in virt])
    lines = ["Experiment 38: projected interaction share on zebrafish groups (Heras et al. 2019)", "",
             f"  {'N':>4}{'rec':>4}{'<phi>':>7}{'synth S':>9}{'synth phi':>10}{'gate':>6}{'S_proj':>8}"
             f"{'alt':>7}{'S_micro':>9}{'loners':>8}{'true':>7}{'res ac':>8}{'align':>8}"]
    gate = {}
    for r in real:
        re, ss, sl = r["real"], r["synthetic_school"], r["synthetic_loners"]
        key = (r["n"], r["rec"])
        gate[key] = ss["S_proj"] >= 0.2 and ss["phi_mean"] >= 0.5 * re["phi_mean"]
        lines.append(f"  {r['n']:4d}{r['rec']:4d}{re['phi_mean']:7.3f}{ss['S_proj']:9.3f}{ss['phi_mean']:10.3f}"
                     f"{'pass' if gate[key] else 'FAIL':>6}{re['S_proj']:8.3f}{re['S_proj_alt']:7.3f}"
                     f"{re['S_micro']:9.3f}{sl['S_proj']:8.3f}{ss['true_twin']['S_proj']:7.3f}"
                     f"{re['res_autocorr']:8.3f}{re['coef'][N_OWN]:8.4f}")
    lines += ["", "  Virtual groups (S_proj, phi, alignment):"]
    for v in virt:
        vv = v["virtual"]
        lines.append(f"    N={v['n']:3d} m={v['m']}: {vv['S_proj']:7.3f}  {vv['phi_mean']:.3f}  {vv['coef'][N_OWN]:.4f}")
    ok = [r for r in real if gate[(r["n"], r["rec"])]]
    tag = lambda v: "PASS" if v else "FAIL"
    counted = lambda f: tag(all(f(r) for r in ok)) if ok else "n/a "
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(all(gate.values()))}  G  gate [{sum(gate.values())}/9 recordings pass]",
              f"  {tag(all(abs(r['synthetic_loners']['S_proj']) <= 0.03 for r in real))}  Z1 synthetic loners |S_proj| <= 0.03 "
              f"[max {max(abs(r['synthetic_loners']['S_proj']) for r in real):.3f}]",
              f"  {tag(all(abs(r['synthetic_school']['S_proj'] - r['synthetic_school']['true_twin']['S_proj']) <= 0.05 for r in real))}  "
              f"Z2 synthetic school: fitted twin within 0.05 of true twin",
              f"  {tag(all(v['virtual']['S_proj'] <= 0.1 for v in virt))}  Z3 virtual groups S_proj <= 0.1 "
              f"[max {max(v['virtual']['S_proj'] for v in virt):.3f}]",
              f"  {counted(lambda r: r['real']['S_proj'] >= vmean[r['n']] + 0.2)}  Z4 real >= virtual mean + 0.2 (gated)",
              f"  {counted(lambda r: abs(r['real']['S_proj'] - r['real']['S_proj_alt']) <= 0.05)}  Z5 state dependence <= 0.05 (gated)",
              f"  {tag(all(r['real']['coef'][N_OWN] > 0 and r['real']['coef'][N_OWN] > 3 * valign for r in real))}  "
              f"Z6 alignment > 0 and > 3x mean |virtual| ({valign:.4f})",
              f"  {tag(all(abs(r['real']['res_autocorr']) <= 0.2 for r in real))}  Z7 |residual autocorrelation| <= 0.2 "
              f"[max {max(abs(r['real']['res_autocorr']) for r in real):.3f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
