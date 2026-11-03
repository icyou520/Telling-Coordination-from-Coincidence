"""Experiment 60: is a level open when there is any slow internal direction its summary misses?

e59 tested the slaving idea with a variance-weighted memory time and failed (2/6): the measure was dominated
by fast, high-variance fluctuations (oscillator noise, fish tracking noise) and could not see slow,
low-variance directions, such as a drifting oscillator's phase. e60 asks the question e59 should have asked,
with slow feature analysis (SFA), which finds the slowest directions regardless of variance.

Residual internal state: the arrangement features (as e59) minus their best linear prediction from the
group's summary block (fitted part). Standardized; principal components with eigenvalue >= 1e-3 of the
largest kept and whitened; SFA directions = eigenvectors of the covariance of one-step changes (fitted part,
consecutive rows only), slowest first. Slow memory time tau_slow: the lag where the slowest direction's
autocorrelation on the measured part falls below 1/e (interpolated in log lag; censored at 30 time units).
No target enters any of this.

Data (new seeds; the same draws as e57 / e58 / e59): 30 oscillator groups (seeds 5000 + i, null 6000 + i;
target r), 24 flocks (seeds 500 + i, null 600 + i; target the turn), the 15 real fish groups (target the
turn; their G was seen before). Test: e56's closure gap G at one time unit with all members' arrangement
features; two extra models use only the 3 slowest or the 3 fastest SFA directions.

Predictions, fixed before running:
  U0  The null is clean (|G| <= 0.01) in >= 80% of the 69 groups.
  U1  Oscillators: Spearman(tau_slow, G(r)) >= 0.6.
  U2  Flocks: Spearman(tau_slow, G(turn)) >= 0.6.
  U3  Fish: Spearman(tau_slow, G(turn)) >= 0.5.
  U4  Pooled: Spearman(tau_slow / horizon, G) >= 0.5.
  U5  No slow direction, closed: groups with tau_slow < 0.3 horizons have G <= 0.02 in >= 80% (if >= 3).
  U6  The information is in the slow directions: in open groups (G >= 0.03, lower > 0),
      G(3 slowest) > G(3 fastest) in >= 80%.
"""

from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e49_school_layer import LAGS, school_state
from experiments.e50_school_configuration import _wrap
from experiments.e53_influence_replication import _rows, place_map
from experiments.e54_neutral_directions import _lags, _quad, _rates, _spearman
from experiments.e56_measured_lock import _closure, influence_sets
from experiments.e58_innovation import _fish, _flock
from experiments.e59_slaving import N_FLOCK, N_OSC, _osc, tasks

SEED = 60
STEPS = 25000
LAG_STEPS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120, 200, 300)
CAP = 30.0
N_DIR = 3


def slow_features(feats, block, train, test, dt):
    """SFA directions of the residual internal state (slowest first), tau_slow and the slowest one's ACF."""
    x1 = np.column_stack([np.ones(len(block)), block])
    xi = feats - x1 @ np.linalg.lstsq(x1[train], feats[train], rcond=None)[0]
    mu, sd = xi[train].mean(0), xi[train].std(0)
    keep = sd > 1e-9
    xi = (xi[:, keep] - mu[keep]) / sd[keep]
    ev, vec = np.linalg.eigh(np.cov(xi[train].T))
    ok = ev >= 1e-3 * ev.max()
    w = xi @ (vec[:, ok] / np.sqrt(ev[ok]))
    tr = train[np.isin(train + 1, train)]
    dv, dvec = np.linalg.eigh(np.cov((w[tr + 1] - w[tr]).T))
    s = w @ dvec
    s1 = s[:, 0]
    acf = []
    for lag in LAG_STEPS:
        te = test[np.isin(test + lag, test)]
        acf.append(float(np.corrcoef(s1[te], s1[te + lag])[0, 1]) if len(te) > 50 else float("nan"))
    tau = CAP
    for j in range(1, len(LAG_STEPS)):
        if acf[j] < np.exp(-1) <= acf[j - 1]:
            l0, l1 = np.log(LAG_STEPS[j - 1]), np.log(LAG_STEPS[j])
            f = (acf[j - 1] - np.exp(-1)) / (acf[j - 1] - acf[j])
            tau = float(np.exp(l0 + f * (l1 - l0)) * dt)
            break
    if acf[0] < np.exp(-1):
        tau = float(LAG_STEPS[0] * dt * max(acf[0], 0) / np.exp(-1))
    return s, min(tau, CAP), acf, float(dv[0] / 2), int(ok.sum())


def _data(family, key):
    if family == "oscillators":
        block, feats, psi, r = _osc(key, 5000 + key)
        null = _osc(key, 6000 + key)[1][0]
        dt, h = 0.1, 10
        rows = np.arange(LAGS + 7, STEPS - h)
        cut = int(0.7 * STEPS)
        train, test = rows[rows + h < cut], rows[rows >= cut]
        y = r[:, 0]
        return block, feats[0], null, y, False, train, test, dt, h
    if family == "flocks":
        ref = np.arange(int(0.7 * STEPS))
        _, p, th, sp, _, _ = _flock(key, 500 + key, ref)
        _, po, tho, spo, _, _ = _flock(key, 600 + key, ref)
        z = school_state(p, th, sp)[:, 2:]
        psi = np.arctan2(z[:, 1], z[:, 0])
        block = np.concatenate([_lags(z), _quad(z), _rates(psi)], 1)
        dt, h = 0.1, 10
        rows = np.arange(LAGS + 7, STEPS - h)
        cut = int(0.7 * STEPS)
        train, test = rows[rows + h < cut], rows[rows >= cut]
    else:
        spec, null_spec = key
        p, th, sp, brk, dt, _ = _fish(spec)
        po, tho, spo, _, _, _ = _fish(null_spec)
        ref = np.arange(int(0.7 * len(th)))
        z = school_state(p, th, sp)
        psi = np.arctan2(z[:, 3], z[:, 2])
        block = np.concatenate([_lags(z), _quad(z), place_map(z, ref), _rates(psi)], 1)
        h = int(round(1 / dt))
        train, test = _rows(brk, len(th), h)
    feats = influence_sets(p, th, sp, ref, [np.arange(th.shape[1])])[0]
    null = np.resize(influence_sets(po, tho, spo, np.arange(len(tho)), [np.arange(tho.shape[1])])[0], feats.shape)
    return block, feats, null, psi, True, train, test, dt, h


def _run(args):
    family, key = args
    rng = np.random.default_rng(SEED * 1000 + zlib.crc32(repr(args).encode()) % 997)
    block, feats, null, y, angular, train, test, dt, h = _data(family, key)
    s, tau, acf, slowness, dims = slow_features(feats, block, train, test, dt)
    rows = np.concatenate([train, test])
    target = (_wrap(y[rows + h] - y[rows]) if angular else y[rows + h] - y[rows])[:, None]
    extras = {"all": feats, "slow": s[:, :N_DIR], "fast": s[:, -N_DIR:], "null": null}
    out = _closure(block, extras, {"y": target}, train, test, h + LAGS, rng, "y", None)["y"]
    name = key if family != "fish" else f"{key[0][0]} {'-'.join(map(str, key[0][1:]))}"
    return {"family": family, "key": name, "dt": float(dt), "h_time": float(h * dt), "tau": tau, "acf": acf,
            "slowness": slowness, "dims": dims, **out}


def _sel(res, family):
    return [r for r in res["runs"] if r["family"] == family]


def predictions(res):
    runs = res["runs"]
    g = lambda r, k="all": r[k]["G"]  # noqa: E731
    nul = sum(abs(g(r, "null")) <= 0.01 for r in runs)
    rho = {f: _spearman([r["tau"] for r in _sel(res, f)], [g(r) for r in _sel(res, f)])
           for f in ("oscillators", "flocks", "fish")}
    pooled = _spearman([r["tau"] / r["h_time"] for r in runs], [g(r) for r in runs])
    fast = [r for r in runs if r["tau"] < 0.3 * r["h_time"]]
    closed = sum(g(r) <= 0.02 for r in fast)
    op = [r for r in runs if g(r) >= 0.03 and r["all"]["lo"] > 0]
    win = sum(g(r, "slow") > g(r, "fast") for r in op)
    return [
        ("U0", "the null is clean in >= 80% of groups", nul >= 0.8 * len(runs), f"{nul}/{len(runs)}"),
        ("U1", "oscillators: Spearman(tau_slow, G(r)) >= 0.6", rho["oscillators"] >= 0.6, f"rho {rho['oscillators']:+.2f}"),
        ("U2", "flocks: Spearman(tau_slow, G(turn)) >= 0.6", rho["flocks"] >= 0.6, f"rho {rho['flocks']:+.2f}"),
        ("U3", "fish: Spearman(tau_slow, G(turn)) >= 0.5", rho["fish"] >= 0.5, f"rho {rho['fish']:+.2f}"),
        ("U4", "pooled: Spearman(tau_slow / horizon, G) >= 0.5", pooled >= 0.5, f"rho {pooled:+.2f}"),
        ("U5", "no slow direction, closed: tau_slow < 0.3 horizons -> G <= 0.02 (>= 80%, if >= 3 groups)",
         len(fast) >= 3 and closed >= 0.8 * len(fast), f"{closed}/{len(fast)}"),
        ("U6", "information in the slow directions: G(3 slowest) > G(3 fastest) in open groups (>= 80%)",
         bool(op) and win >= 0.8 * len(op), f"{win}/{len(op)}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 60: is a level open when there is any slow internal direction its summary misses?", ""]
    for f in ("oscillators", "flocks", "fish"):
        for r in sorted(_sel(res, f), key=lambda r: r["tau"]):
            lines.append(f"  {f:11} {str(r['key']):15} tau_slow {r['tau']:6.2f} (dims {r['dims']:2d}) | G all "
                         f"{r['all']['G']:+.3f} [{r['all']['lo']:+.3f}]  slow-3 {r['slow']['G']:+.3f}  fast-3 "
                         f"{r['fast']['G']:+.3f}  null {r['null']['G']:+.3f}")
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
    jobs = tasks()
    if quick:
        jobs = jobs[:2] + jobs[N_OSC:N_OSC + 2] + jobs[-1:]
    assert len(jobs) == N_OSC + N_FLOCK + 15 or quick
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run, jobs))
    res = {"runs": runs, "lags": list(LAG_STEPS)}
    (out_dir / "e60_slow_features_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e60_figures import make_figures
    make_figures(res, out_dir / "e60_figures", text)
    return text


if __name__ == "__main__":
    main()
