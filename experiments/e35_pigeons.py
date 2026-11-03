"""Experiment 35: the projected interaction share on pigeon flocks (no walls).

Second real data set, chosen because it has no walls: GPS tracks of three
flocks (A, B, C) of 10 homing pigeons on 5 free flights each around their
loft (Nagy et al. 2013). Only the file format, the timing and the extent of
the flights were inspected before these predictions were written: the birds
circle about 50 m from the loft at about 12 m/s, and all ten fly together
for 1.6-17 min per flight, in blocks of 1-6 min.

Pipeline (method of e33, adapted; all choices fixed in advance):

  Data      native 0.2 s GPS samples where all birds fly (> 5 m/s); a time
            step is never taken across a gap. Flights of a flock are pooled;
            the first 70 % of valid macro pairs (in flight order) train, the
            rest test.
  Macro     polarization phi, macro step 1 s (5 samples).
  Micro     least squares on the 0.2 s heading change:
              own terms   intercept; homing sin(beta - theta) * (1, r / 50 m,
                          (r / 50 m)^2) with beta the bearing to the loft and r
                          its distance; the bird's own last 3 turns (the
                          tracks are smoothed, so persistence is strong)
              social      alignment, attraction and repulsion
                          (exp(-d_nn / 2 m)) with the 4 nearest birds, as e31
  Twin      own terms + resampled residuals; each bird keeps its speed.
  S_proj    G cubic in phi on training pairs; G0_proj cubic fit of the
            micro-informed twin prediction (64 replicas) against phi on
            training states; scored on test pairs.
  Control   virtual flocks: within each flight, bird i's flying samples are
            circularly shifted by i / N of their length (>= 10 s apart), so
            the birds circle the same loft but never together; steps that
            cross a gap in any bird's source series are dropped.
  Synthetic the full model fitted to all of a flock's data, simulated as a "flock" and as
            "loners" (twin), for as many samples as the flock has, started
            from its first state; the pipeline is run on these as if data.

Predictions, fixed before running (each for flocks A, B and C):

  Q1  Synthetic loners: |S_proj| <= 0.03.
  Q2  Synthetic flock: S_proj from the fitted twin within 0.05 of the true twin's.
  Q3  Virtual flocks: S_proj <= 0.1.
  Q4  Real flocks: S_proj >= S_proj(virtual) + 0.2.
  Q5  Alignment > 0 and > 3x the virtual flock's (in absolute value).
  Q6  State dependence: fitting G0_proj on test instead of training states
      changes real S_proj by <= 0.05.
Exploratory: S_micro, residual autocorrelation, macro-pair counts.

Fixed before any result was produced: the first run crashed because flight
ffA1 has only 8 tracked birds and cannot be pooled with 10-bird flights; it
is excluded (flock A uses 4 flights).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import _wrap, polarization, social_terms
from experiments.pigeon_data import DT, FLOCKS, flights, load

TAU = 5
LAGS = 3
STRIDE = 2
REPLICAS = 64
MAX_STATES = 1500
TRAIN = 0.7
N_OWN = 4 + LAGS


def configure(dt: float, tau: int, lags: int, stride: int) -> None:
    global DT, TAU, LAGS, STRIDE, N_OWN
    DT, TAU, LAGS, STRIDE, N_OWN = dt, tau, lags, stride, 4 + lags


def homing(p, theta):
    r = np.hypot(p[..., 0], p[..., 1]) / 50.0
    s = np.sin(np.arctan2(-p[..., 1], -p[..., 0]) - theta)
    return np.stack([s, s * r, s * r * r], -1)


def own_turn(p, theta, lags, c):
    return c[0] + homing(p, theta) @ c[1:4] + lags @ c[4:4 + LAGS]


def prepare(series):
    """Add turn lags (NaN where a gap intervenes) and the target turn."""
    th, cont = series["theta"], series["cont"]
    turn_in = np.full(th.shape, np.nan)
    turn_in[1:] = np.where(cont[1:, None], _wrap(th[1:] - th[:-1]), np.nan)
    lags = np.full(th.shape + (LAGS,), np.nan)
    for k in range(LAGS):
        lags[k:, :, k] = turn_in[:len(th) - k]
    target = np.full(th.shape, np.nan)
    target[:-1] = turn_in[1:]
    return dict(series, lags=lags, target=target)


def fit(all_series):
    xs, ys = [], []
    for s in all_series:
        stop = s["train_stop"]
        own = np.concatenate([np.ones(s["theta"].shape + (1,)), homing(s["p"], s["theta"]),
                              s["lags"]], -1)
        x = np.concatenate([own, social_terms(s["p"], s["theta"])], -1)[:stop]
        y = s["target"][:stop]
        ok = np.isfinite(y) & np.isfinite(x).all(-1)
        xs.append(x[ok]), ys.append(y[ok])
    x, y = np.concatenate(xs), np.concatenate(ys)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return coef, y - x @ coef


def macro_pairs(s):
    """Times t with a valid 1 s macro step and finite lags."""
    ok = np.isfinite(s["lags"]).all((1, 2))
    good = []
    for t in range(len(s["theta"]) - TAU):
        if ok[t] and s["cont"][t + 1:t + TAU + 1].all():
            good.append(t)
    return np.array(good, int)


def twin_predict(p, theta, speed, lags, c, res, rng):
    p, th, sp, lg = (np.repeat(a[:, None], REPLICAS, 1) for a in (p, theta, speed, lags))
    for _ in range(TAU):
        turn = own_turn(p, th, lg, c) + rng.choice(res, th.shape)
        th = _wrap(th + turn)
        p = p + (sp * DT)[..., None] * np.stack([np.cos(th), np.sin(th)], -1)
        lg = np.concatenate([turn[..., None], lg[..., :-1]], -1)
    return polarization(th).mean(-1)


def _sub(idx):
    return idx if len(idx) <= MAX_STATES else idx[np.linspace(0, len(idx) - 1, MAX_STATES).astype(int)]


def analyse(raw_series, rng, true_twin=None):
    series = [prepare(s) for s in raw_series]
    pairs = [macro_pairs(s) for s in series]
    total = sum(len(pp) for pp in pairs)
    cut = int(TRAIN * total)
    seen = 0
    for s, pp in zip(series, pairs):
        k = int(np.clip(cut - seen, 0, len(pp)))
        s["train_stop"] = int(pp[k - 1]) + TAU + 1 if k > 0 else 0
        s["is_train"] = np.arange(len(pp)) < k
        seen += len(pp)
    coef, res = fit(series)
    stacked = {key: np.concatenate([s[key][pp] for s, pp in zip(series, pairs)])
               for key in ("p", "theta", "speed", "lags")}
    x = np.concatenate([polarization(s["theta"][pp]) for s, pp in zip(series, pairs)])
    z = np.concatenate([polarization(s["theta"][pp + TAU]) for s, pp in zip(series, pairs)])
    is_train = np.concatenate([s["is_train"] for s in series])
    tr, te = _sub(np.flatnonzero(is_train)), _sub(np.flatnonzero(~is_train))
    g = np.polyval(np.polyfit(x[is_train], z[is_train], 3), x[te])
    mse = lambda pred: float(np.mean((z[te] - pred) ** 2))

    def shares(c, r):
        g0 = lambda idx: twin_predict(*(stacked[k][idx] for k in ("p", "theta", "speed", "lags")),
                                      c, r, rng)
        g0_tr, g0_te = g0(tr), g0(te)
        return {"S_proj": 1 - mse(g) / mse(np.polyval(np.polyfit(x[tr], g0_tr, 3), x[te])),
                "S_proj_alt": 1 - mse(g) / mse(np.polyval(np.polyfit(x[te], g0_te, 3), x[te])),
                "S_micro": 1 - mse(g) / mse(g0_te)}

    out = {"coef": coef.tolist(), "pairs_train": int(is_train.sum()), "pairs_test": int((~is_train).sum()),
           "phi_mean": float(x.mean()), "res_sd": float(res.std()), **shares(coef[:N_OWN], res)}
    if true_twin is not None:
        out["true_twin"] = shares(*true_twin)
    return out


def virtual(s):
    t, n = s["theta"].shape
    src = np.stack([(np.arange(t) + i * t // n) % t for i in range(n)], 1)
    cont = np.zeros(t, bool)
    cont[1:] = ((src[1:] == src[:-1] + 1) & s["cont"][src[1:]]).all(1)
    cols = np.arange(n)
    return {"p": s["p"][src, cols], "theta": s["theta"][src, cols],
            "speed": s["speed"][src, cols], "cont": cont}


def simulate(coef, res, speed, p0, th0, steps, social, rng):
    n = len(speed)
    p, th, lg = p0.copy(), th0.copy(), np.zeros((n, LAGS))
    ps, ths = np.empty((steps, n, 2)), np.empty((steps, n))
    for t in range(steps):
        ps[t], ths[t] = p, th
        turn = own_turn(p, th, lg, coef) + rng.choice(res, n)
        if social:
            turn = turn + social_terms(p[None], th[None])[0] @ coef[N_OWN:]
        th = _wrap(th + turn)
        p = p + (speed * DT)[:, None] * np.stack([np.cos(th), np.sin(th)], -1)
        lg = np.concatenate([turn[:, None], lg[:, :-1]], -1)
    return {"p": ps, "theta": ths, "speed": np.broadcast_to(speed, ths.shape).copy(),
            "cont": np.r_[False, np.ones(steps - 1, bool)]}


def _flock(flock):
    rng = np.random.default_rng(35_000 + ord(flock))
    real = [load(f, STRIDE) for f in flights(flock)]
    out = {"flock": flock, "minutes": sum(len(s["cont"]) for s in real) * DT / 60}
    out["real"] = analyse(real, rng)
    out["virtual"] = analyse([virtual(s) for s in real], rng)
    series = [prepare(s) for s in real]
    for s in series:
        s["train_stop"] = len(s["theta"])
    coef, res = fit(series)
    speed = np.median(np.concatenate([s["speed"] for s in real]), 0)
    steps = sum(len(s["cont"]) for s in real)
    for kind, social in (("flock", True), ("loners", False)):
        sim = simulate(coef, res, speed, real[0]["p"][0], real[0]["theta"][0], steps, social, rng)
        out["synthetic_" + kind] = analyse([sim], rng, true_twin=(coef[:N_OWN], res))
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
    (out_dir / "e35_pigeons_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    lines = ["Experiment 35: projected interaction share on pigeon flocks (Nagy et al. 2013)", "",
             f"  {'flock':>5}{'min':>6}{'pairs':>11}{'<phi>':>7}{'S_proj':>8}{'virtual':>9}"
             f"{'S_micro':>9}{'alt':>7}{'align':>8}{'virt.':>8}{'synth flock':>13}{'true':>7}"
             f"{'loners':>8}"]
    for r in rows:
        re, vi, sf, sl = r["real"], r["virtual"], r["synthetic_flock"], r["synthetic_loners"]
        lines.append(f"  {r['flock']:>5}{r['minutes']:6.1f}{re['pairs_train']:>6}/{re['pairs_test']:<4}"
                     f"{re['phi_mean']:7.3f}{re['S_proj']:8.3f}{vi['S_proj']:9.3f}{re['S_micro']:9.3f}"
                     f"{re['S_proj_alt']:7.3f}{re['coef'][N_OWN]:8.4f}{vi['coef'][N_OWN]:8.4f}"
                     f"{sf['S_proj']:13.3f}{sf['true_twin']['S_proj']:7.3f}{sl['S_proj']:8.3f}")
    q1 = all(abs(r["synthetic_loners"]["S_proj"]) <= 0.03 for r in rows)
    q2 = all(abs(r["synthetic_flock"]["S_proj"] - r["synthetic_flock"]["true_twin"]["S_proj"]) <= 0.05
             for r in rows)
    q3 = all(r["virtual"]["S_proj"] <= 0.1 for r in rows)
    q4 = all(r["real"]["S_proj"] >= r["virtual"]["S_proj"] + 0.2 for r in rows)
    q5 = all(r["real"]["coef"][N_OWN] > 0 and r["real"]["coef"][N_OWN] > 3 * abs(r["virtual"]["coef"][N_OWN])
             for r in rows)
    q6 = max(abs(r["real"]["S_proj"] - r["real"]["S_proj_alt"]) for r in rows)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(q1)}  Q1 synthetic loners |S_proj| <= 0.03",
              f"  {tag(q2)}  Q2 synthetic flock: fitted twin within 0.05 of true twin",
              f"  {tag(q3)}  Q3 virtual flocks S_proj <= 0.1",
              f"  {tag(q4)}  Q4 real S_proj >= virtual + 0.2",
              f"  {tag(q5)}  Q5 alignment > 0 and > 3x virtual",
              f"  {tag(q6 <= 0.05)}  Q6 state dependence <= 0.05 [max {q6:.3f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
