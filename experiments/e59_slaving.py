"""Experiment 59: is a level open when its internal state is slow? (slaving, as a test)

e55-e58 suggested one idea for oscillators, flocks and fish together: a level is open when some internal
degrees of freedom are slow, persisting over the forecast horizon instead of relaxing quickly to follow the
group's summary (Haken's slaving principle). Locked oscillators relax fast and the group closes; drifting
ones persist and it opens. A flock's arrangement persists about one time unit and its turn is open.

Internal memory time tau: the members' arrangement features (e54 phase features for oscillators, e52's 75
influence features for flocks and fish), minus their best linear prediction from the group's summary
block (fitted part only), standardized, reduced to the principal components that hold 90% of the variance
(at most 10). For lags s, a linear map from the residual at t to t + s is fitted on the fitted part and
scored on the measured part (R^2). tau is the lag where R^2 falls below 1/e (interpolated in log s;
censored at 30 time units). No target enters tau.

Data (new runs; the same parameter draws as e57 and e58, other seeds):
  oscillators  30 groups, e57's draws, seeds 3000 + i; target r (e55-e57)
  flocks       24 flocks, e58's draws (1-4 informed agents), seeds 300 + i; target the turn
  fish         the 6 Puy et al. schools and 9 zebrafish groups; target the turn (their G was seen before)
Test: e56's closure gap G at one time unit (1 s) with all members' arrangement features (e57's loosest-3
reported for oscillators). Null: the arrangement of another run / group.

Predictions, fixed before running:
  T0  The null is clean (|G| <= 0.01) in >= 80% of all 69 groups.
  T1  Oscillators: Spearman(tau, G(r)) >= 0.6.
  T2  Flocks: Spearman(tau, G(turn)) >= 0.6.
  T3  Fish: Spearman(tau, G(turn)) >= 0.5.
  T4  Pooled over all 69 groups: Spearman(tau / horizon, G) >= 0.5.
  T5  Fast inside, closed outside: groups with tau < 0.3 horizons have G <= 0.02 in >= 80% (if any).
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
from experiments.e58_innovation import tasks as fish_tasks

SEED = 59
N_OSC, N_FLOCK = 30, 24
STEPS = 25000
LAG_STEPS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120, 200, 300)
CAP = 30.0


def memory_time(feats, block, train, test, dt):
    """Internal memory time (time units) and the R^2 curve over LAG_STEPS."""
    x1 = np.column_stack([np.ones(len(block)), block])
    xi = feats - x1 @ np.linalg.lstsq(x1[train], feats[train], rcond=None)[0]
    mu, sd = xi[train].mean(0), xi[train].std(0)
    keep = sd > 1e-9
    xi = (xi[:, keep] - mu[keep]) / sd[keep]
    ev, vec = np.linalg.eigh(np.cov(xi[train].T))
    ev, vec = ev[::-1], vec[:, ::-1]
    q = int(min(10, np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9) + 1))
    z = xi @ vec[:, :q]
    r2 = []
    for s in LAG_STEPS:
        tr, te = train[train + s <= train[-1]], test[test + s < len(z)]
        a = np.column_stack([np.ones(len(tr)), z[tr]])
        coef = np.linalg.lstsq(a, z[tr + s], rcond=None)[0]
        e = z[te + s] - np.column_stack([np.ones(len(te)), z[te]]) @ coef
        r2.append(float(1 - (e ** 2).sum() / ((z[te + s] - z[te + s].mean(0)) ** 2).sum()))
    tau = CAP
    for j in range(1, len(LAG_STEPS)):
        if r2[j] < np.exp(-1) <= r2[j - 1]:
            l0, l1 = np.log(LAG_STEPS[j - 1]), np.log(LAG_STEPS[j])
            f = (r2[j - 1] - np.exp(-1)) / (r2[j - 1] - r2[j])
            tau = float(np.exp(l0 + f * (l1 - l0)) * dt)
            break
    if r2[0] < np.exp(-1):
        tau = float(LAG_STEPS[0] * dt * r2[0] / np.exp(-1))
    return min(tau, CAP), r2, q


def _osc(i, seed):
    from experiments.e57_oscillator_looseness import _sim
    cut = int(0.7 * STEPS)
    lam, block, feats, psi, r = _sim(i, seed, np.arange(cut))
    return block, feats, psi, r


def _split(T, h, brk=None):
    rows = np.arange(LAGS + 7, T - h)
    if brk is not None:
        return _rows(brk, T, h)
    cut = int(0.7 * T)
    return rows[rows + h < cut], rows[rows >= cut]


def _run(args):
    family, key = args
    rng = np.random.default_rng(SEED * 1000 + zlib.crc32(repr(args).encode()) % 997)
    if family == "oscillators":
        block, feats, psi, r = _osc(key, 3000 + key)
        null = _osc(key, 4000 + key)[1][0]
        dt, h = 0.1, 10
        train, test = _split(STEPS, h)
        rows = np.concatenate([train, test])
        target = r[rows + h, 0] - r[rows, 0]
        extras = {"all": feats[0], "loosest": feats[1], "null": null}
        x_tau = feats[0]
    else:
        if family == "flocks":
            ref = np.arange(int(0.7 * STEPS))
            _, p, th, sp, _, _ = _flock(key, 300 + key, ref)
            _, po, tho, spo, _, _ = _flock(key, 400 + key, ref)
            z = school_state(p, th, sp)[:, 2:]
            psi = np.arctan2(z[:, 1], z[:, 0])
            block = np.concatenate([_lags(z), _quad(z), _rates(psi)], 1)
            dt, h = 0.1, 10
            train, test = _split(STEPS, h)
        else:
            spec, null_spec = key
            p, th, sp, brk, dt, _ = _fish(spec)
            po, tho, spo, _, _, _ = _fish(null_spec)
            ref = np.arange(int(0.7 * len(th)))
            z = school_state(p, th, sp)
            psi = np.arctan2(z[:, 3], z[:, 2])
            block = np.concatenate([_lags(z), _quad(z), place_map(z, ref), _rates(psi)], 1)
            h = int(round(1 / dt))
            train, test = _split(len(th), h, brk)
        feats = influence_sets(p, th, sp, ref, [np.arange(th.shape[1])])[0]
        null = np.resize(influence_sets(po, tho, spo, np.arange(len(tho)), [np.arange(tho.shape[1])])[0], feats.shape)
        rows = np.concatenate([train, test])
        target = _wrap(psi[rows + h] - psi[rows])
        extras = {"all": feats, "null": null}
        x_tau = feats
    tau, r2, q = memory_time(x_tau, block, train, test, dt)
    out = _closure(block, extras, {"y": target[:, None]}, train, test, h + LAGS, rng, "y", None)["y"]
    name = key if family != "fish" else f"{key[0][0]} {'-'.join(map(str, key[0][1:]))}"
    return {"family": family, "key": name, "dt": float(dt), "h_time": float(h * dt), "tau": tau, "r2": r2,
            "q": q, **{k: v for k, v in out.items()}}


def _sel(res, family):
    return [r for r in res["runs"] if r["family"] == family]


def predictions(res):
    runs = res["runs"]
    nul = sum(abs(r["null"]["G"]) <= 0.01 for r in runs)
    rho = {f: _spearman([r["tau"] for r in _sel(res, f)], [r["all"]["G"] for r in _sel(res, f)])
           for f in ("oscillators", "flocks", "fish")}
    pooled = _spearman([r["tau"] / r["h_time"] for r in runs], [r["all"]["G"] for r in runs])
    fast = [r for r in runs if r["tau"] < 0.3 * r["h_time"]]
    closed = sum(r["all"]["G"] <= 0.02 for r in fast)
    return [
        ("T0", "the null is clean in >= 80% of groups", nul >= 0.8 * len(runs), f"{nul}/{len(runs)}"),
        ("T1", "oscillators: Spearman(tau, G(r)) >= 0.6", rho["oscillators"] >= 0.6, f"rho {rho['oscillators']:+.2f}"),
        ("T2", "flocks: Spearman(tau, G(turn)) >= 0.6", rho["flocks"] >= 0.6, f"rho {rho['flocks']:+.2f}"),
        ("T3", "fish: Spearman(tau, G(turn)) >= 0.5", rho["fish"] >= 0.5, f"rho {rho['fish']:+.2f}"),
        ("T4", "pooled: Spearman(tau / horizon, G) >= 0.5", pooled >= 0.5, f"rho {pooled:+.2f}"),
        ("T5", "fast inside, closed outside: tau < 0.3 horizons -> G <= 0.02 (>= 80%)",
         bool(fast) and closed >= 0.8 * len(fast), f"{closed}/{len(fast)}" if fast else "no such groups"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 59: is a level open when its internal state is slow?", ""]
    for f in ("oscillators", "flocks", "fish"):
        for r in sorted(_sel(res, f), key=lambda r: r["tau"]):
            extra = f"  loosest-3 {r['loosest']['G']:+.3f}" if "loosest" in r else ""
            lines.append(f"  {f:11} {str(r['key']):10} tau {r['tau']:6.2f} (q {r['q']:2d}, R2 at 1 unit "
                         f"{r['r2'][LAG_STEPS.index(10)]:+.2f}) | G {r['all']['G']:+.3f} [{r['all']['lo']:+.3f}]"
                         f"{extra}  null {r['null']['G']:+.3f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def tasks():
    _, fish = fish_tasks()
    return ([("oscillators", i) for i in range(N_OSC)] + [("flocks", i) for i in range(N_FLOCK)]
            + [("fish", f) for f in fish])


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    jobs = tasks()
    if quick:
        jobs = jobs[:2] + jobs[N_OSC:N_OSC + 2] + jobs[-1:]
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run, jobs))
    res = {"runs": runs, "lags": list(LAG_STEPS)}
    (out_dir / "e59_slaving_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e59_figures import make_figures
    make_figures(res, out_dir / "e59_figures", text)
    return text


if __name__ == "__main__":
    main()
