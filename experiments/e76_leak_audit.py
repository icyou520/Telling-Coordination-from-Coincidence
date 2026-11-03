"""Experiment 76: a leak in the cross-fitted projected share, its size, and the memory gap on fresh markets.

e75 left breadth (the share of industries on an extreme day) below its memory-gap prediction in 3 of 5 real sets.
An exploratory decomposition on those (already seen) US sets found the cause, and it is not the equation:
  pooled vs per-member memory, cross-fitting with OLS, day-of-week, ridge: none moved the share by more than 0.008;
  the measured pipeline (e66.share, and e67.measured) feeds the projection out-of-fold twin forecasts. For test
  fold f, the projection is fitted on the other folds' twin forecasts, which come from member models fitted on
  folds that include f. The twin sees the test fold, looks better, and S is biased low. Removing it (member models of
  fold f, fitted on the other folds, applied to every day; projection fitted on those) brought breadth to its
  prediction: equal-weighted 1972-2026 0.015 against 0.016 (was 0.005), 1936-1971 0.026 and 0.027 against 0.024
  and 0.025.
Affected: e64, e65 (silencing, not reanalysed here), e66, e69, e74, e75 (markets), e67's measured share (its
simulations, the 221 brain areas) and everything scored with it (ALM in e70, e71, e73). Not affected: e61/e62 (one
train/test split), the fish twins, and every memory-gap prediction (no fitting). The bias can only make the twin look
better, so affected shares are too low, never too high.

Development (e66 simulations, seeds 100-102, 14,000 and 2,800 days), leaky -> leak-free:
  14,000 days, mean and breadth: every cell rose (by 0.001-0.009); independent markets -0.0024..-0.0006 -> within
  +-0.0011 of 0; leak-free within 0.001 of the memory gap in every non-independent cell.
  2,800 days: rises of 0.004-0.06, leak-free still noisy (one factor seed 0.064 against 0.016 predicted).
  e75's volatility twin (max, systemic): differences in both directions, no systematic leak.

Part A, simulations (fresh seeds 20-24, e66 markets, 14,000 days): mean and breadth, leaky and leak-free.
Part B, fresh markets: daily value-weighted returns of 25 size x book-to-market portfolios in Japan, Europe and Asia
Pacific ex Japan, 1990-2026 (9,458 days; French library; never analysed). Members log(|r| + 0.05); mean and breadth;
leak-free share with 400 block-bootstrap resamples and 5 virtual groups; memory gap with HAR windows.
Part C, reanalysis (corrected values never computed): e67's 66 simulations and 221 brain areas, the 38 ALM sessions
of e71 (holdout) and e73 (Economo), each with e67.measured and the leak-free version.

Predictions, fixed before running:
  A1  The leak lowers S: leak-free > leaky in >= 90% of simulated cells (mean and breadth).
  A2  Leak-free is unbiased under independence: |S| <= 0.003, mean and breadth, >= 4/5 seeds each.
  A3  Leak-free matches the memory gap: |S - S_pred| <= 0.003 in >= 90% of non-independent cells.
  A4  The leak shows under independence: leaky mean S < 0 in >= 4/5 seeds.
  B1  Fresh markets: |S - S_pred| <= 0.01 for mean and breadth in all three regions.
  B2  Fresh markets, virtual groups: all |S| <= 0.01, mean and breadth.
  B3  Coordination in volatility abroad: mean S >= 0.005 with 90% lower bound > 0 in >= 2 of 3 regions.
  B4  The leak is larger for rare-event members: (leak-free - leaky) larger for breadth than mean in >= 2 of 3.
  C1  Brain areas: leak-free >= leaky in >= 80% of areas, and Spearman(leak-free S, S_pred) >= 0.9.
  C2  Brain areas: median |S - S_pred| smaller leak-free than leaky.
  C3  ALM: median leak-free S >= median leaky S, and leak-free S > 0 in >= 90% of the 38 sessions.
  C4  e67 simulations: leak-free meets e67's P1 tolerance (|S - S_pred| <= 0.02 + 0.1 S_pred) in >= 90% of the
      linear cells.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments import e66_market_volatility as e66
from experiments import e67_memory_gap as e67
from experiments import e74_nonlinear_gap as e74

SEED = 76


# ---------------------------------------------------------------- leak-free constructions

def share_series(x, dow):
    """e66.share without the leak: for test fold f every member model is fitted on the other folds and applied to
    all days, and the projection is fitted on those forecasts. Returns per-day errors and blocks."""
    days, n = x.shape
    env = np.eye(5)[dow[e66.LAGS:]][:, 1:]
    fold, block = e66.folds(days - e66.LAGS)
    feats, y = e66.har(x), x[e66.LAGS:]
    xg = np.column_stack([e66.har(x.mean(1)), env])
    yg = x.mean(1)[e66.LAGS:]
    e_g, e_0 = np.empty(len(yg)), np.empty(len(yg))
    for f in range(3):
        tr, te = fold != f, fold == f
        y0 = np.mean([e66._ridge(np.column_stack([feats[:, i], env]), y[:, i], tr) for i in range(n)], 0)
        e_g[te] = (yg[te] - e66._ridge(xg, yg, tr)[te]) ** 2
        e_0[te] = (yg[te] - e66._ridge(xg, y0, tr)[te]) ** 2
    return e_g, e_0, block


def measured(x, offs, f, y, rows, fold_of_segment):
    """e67.measured without the leak (same construction, member models of fold k applied to every row)."""
    n, ns = x.shape[:2]
    hist, target = e67.features(x, offs, f, y, rows)
    nr = len(rows)
    fold = np.repeat(fold_of_segment, nr)
    gh, gt = e67.features(x.sum(0), offs, f, y, rows)
    xg, yg = gh.reshape(ns * nr, -1), gt.ravel()
    e_g, e_0 = 0.0, 0.0
    for k in np.unique(fold):
        te = fold == k
        y0 = sum(e67._ridge(hist[i].reshape(ns * nr, -1), target[i].ravel(), ~te) for i in range(n))
        e_g += ((yg[te] - e67._ridge(xg, yg, ~te)[te]) ** 2).sum()
        e_0 += ((yg[te] - e67._ridge(xg, y0, ~te)[te]) ** 2).sum()
    return float(1 - e_g / e_0)


_LEAKY = e67.measured
_LOG: list = []


def _both(x, offs, f, y, rows, fold_of_segment):
    _LOG.append(_LEAKY(x, offs, f, y, rows, fold_of_segment))
    return measured(x, offs, f, y, rows, fold_of_segment)


def _patched(fn, arg):
    """Run an e67-based job with the leak-free share, logging e67's (leaky) share of every call in order."""
    _LOG.clear()
    e67.measured = _both
    try:
        out = fn(arg)
    finally:
        e67.measured = _LEAKY
    return out, list(_LOG)


# ---------------------------------------------------------------- Part A and B: market series

A_SEEDS = tuple(range(20, 25))
CONDS = e66.SIM_CONDITIONS
VARS = ("mean", "breadth")


def _s(e):
    return float(1 - e[0].sum() / e[1].sum())


def series_shares(l, dow, rng, full=True):
    """Leaky (e66.share) and leak-free S for mean log volatility and breadth, with the memory gap."""
    ind = (l > np.quantile(l, e74.TAIL, 0)).astype(float)
    out = {}
    for v, x in zip(VARS, (l, ind)):
        e_g, e_0, block = share_series(x, dow)
        r = e66.summarize(e_g, e_0, block, rng) if full else {"S": _s((e_g, e_0))}
        out[v] = {**r, "leaky": _s(e66.share(x, dow)[:2])}
        if full:
            out[v]["S_pred"] = e74._pred(x)
    return out


def _sim(job):
    cond, seed = job
    l = np.log(e66.simulate(cond, seed) + e74.OFFSET)
    rng = np.random.default_rng([SEED, seed, CONDS.index(cond)])
    return {"cond": cond, "seed": seed, **series_shares(l, e66._dow(len(l)), rng)}


def _region(region):
    from experiments import finance_data as fd
    dates, r, names = fd.regional(region)
    l, dow = np.log(np.abs(r) + e74.OFFSET), e66._dow_dates(dates)
    rng = np.random.default_rng([SEED, len(region)])
    out = {"region": region, "members": len(names), "days": len(dates), **series_shares(l, dow, rng)}
    out["virtual"] = {v: [] for v in VARS}
    for _ in range(e66.VIRTUAL):
        s = series_shares(e66.virtual(l, rng), dow, rng, full=False)
        for v in VARS:
            out["virtual"][v].append(s[v]["S"])
    return out


# ---------------------------------------------------------------- Part C: reanalysis of e67's measured share

def _c_linear(job):
    out, log = _patched(e67._job_linear, job)
    return {**out, "leaky": log[0]}


def _c_garch(job):
    out, log = _patched(e67._job_garch, job)
    return {**out, "leaky": log[0]}


def _c_spiking(job):
    out, log = _patched(e67._job_spiking, job)
    return {**out, "leaky": log[0]}


def _c_area(s):
    out, log = _patched(e67._job_area, s)
    return [{**a, "leaky": v} for a, v in zip(out, log)]


def _c_alm(job):
    from experiments import e71_alm_holdout as e71, e73_alm_replication as e73
    study, name = job
    out, log = _patched(e71._holdout if study == "holdout" else e73._replicate, name)
    return {"study": study, **out, "session": name, "leaky": log[0], "virtual_leaky": log[1], "drift_leaky": log[2]}


def alm_sessions():
    return ([("holdout", r["session"]) for r in json.loads(Path("results/e71_alm_holdout_raw.json").read_text())["holdout"]]
            + [("replication", r["session"])
               for r in json.loads(Path("results/e73_alm_replication_raw.json").read_text())["replication"]])


# ---------------------------------------------------------------- predictions

def _tol67(r):
    return abs(r["S"] - r["S_pred"]) <= 0.02 + 0.1 * abs(r["S_pred"])


def predictions(res):
    sim = res["sim"]
    cells = [r[v] for r in sim for v in VARS]
    a1 = sum(c["S"] > c["leaky"] for c in cells)
    ind = [r for r in sim if r["cond"] == "independent"]
    a2 = {v: sum(abs(r[v]["S"]) <= 0.003 for r in ind) for v in VARS}
    dep = [r[v] for r in sim if r["cond"] != "independent" for v in VARS]
    a3 = sum(abs(c["S"] - c["S_pred"]) <= 0.003 for c in dep)
    a4 = sum(r["mean"]["leaky"] < 0 for r in ind)
    reg = res["regions"]
    b1 = [abs(r[v]["S"] - r[v]["S_pred"]) for r in reg for v in VARS]
    b2 = [abs(s) for r in reg for v in VARS for s in r["virtual"][v]]
    b3 = sum(r["mean"]["S"] >= 0.005 and r["mean"]["lo"] > 0 for r in reg)
    b4 = sum(r["breadth"]["S"] - r["breadth"]["leaky"] > r["mean"]["S"] - r["mean"]["leaky"] for r in reg)
    areas = res["areas"]
    c1a = sum(a["S"] >= a["leaky"] for a in areas)
    c1b = e67._spearman([a["S"] for a in areas], [a["S_pred"] for a in areas])
    c2 = (float(np.median([abs(a["S"] - a["S_pred"]) for a in areas])),
          float(np.median([abs(a["leaky"] - a["S_pred"]) for a in areas])))
    alm = res["alm"]
    c3 = (float(np.median([r["S"] for r in alm])), float(np.median([r["leaky"] for r in alm])),
          sum(r["S"] > 0 for r in alm))
    lin = [r for r in res["c_sim"] if r["kind"] in ("factor", "meanfield", "local")]
    c4 = sum(_tol67(r) for r in lin)
    c4_old = sum(_tol67({**r, "S": r["leaky"]}) for r in lin)
    nr, nd = len(reg), len(dep)
    return [
        ("A1", "the leak lowers S: leak-free > leaky in >= 90% of simulated cells", a1 >= 0.9 * len(cells),
         f"{a1}/{len(cells)}"),
        ("A2", "leak-free unbiased under independence: |S| <= 0.003 in >= 4/5 seeds, mean and breadth",
         all(k >= 0.8 * len(ind) for k in a2.values()), f"mean {a2['mean']}/{len(ind)}, breadth {a2['breadth']}/{len(ind)}"),
        ("A3", "leak-free matches the memory gap: |S - S_pred| <= 0.003 in >= 90% of non-independent cells",
         a3 >= 0.9 * nd, f"{a3}/{nd}"),
        ("A4", "the leak shows under independence: leaky mean S < 0 in >= 4/5 seeds", a4 >= 0.8 * len(ind),
         f"{a4}/{len(ind)}"),
        ("B1", "fresh markets: |S - S_pred| <= 0.01 for mean and breadth in all three regions",
         max(b1) <= 0.01, f"max |diff| {max(b1):.4f}"),
        ("B2", "fresh markets, virtual groups: all |S| <= 0.01", max(b2) <= 0.01, f"max |S| {max(b2):.4f}"),
        ("B3", "coordination abroad: mean S >= 0.005 with lower bound > 0 in >= 2 of 3 regions", b3 >= 2,
         f"{b3}/{nr}"),
        ("B4", "the leak is larger for breadth than for the mean in >= 2 of 3 regions", b4 >= 2, f"{b4}/{nr}"),
        ("C1", "brain areas: leak-free >= leaky in >= 80%, Spearman(S, S_pred) >= 0.9",
         c1a >= 0.8 * len(areas) and c1b >= 0.9, f"{c1a}/{len(areas)}, rho {c1b:+.3f}"),
        ("C2", "brain areas: median |S - S_pred| smaller leak-free than leaky", c2[0] < c2[1],
         f"{c2[0]:.4f} vs {c2[1]:.4f}"),
        ("C3", "ALM: median leak-free S >= median leaky, leak-free S > 0 in >= 90% of sessions",
         c3[0] >= c3[1] and c3[2] >= 0.9 * len(alm), f"median {c3[0]:+.4f} vs {c3[1]:+.4f}; {c3[2]}/{len(alm)} > 0"),
        ("C4", "e67 simulations: leak-free meets P1's tolerance in >= 90% of linear cells", c4 >= 0.9 * len(lin),
         f"{c4}/{len(lin)} (leaky {c4_old}/{len(lin)})"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 76: a leak in the cross-fitted projected share, its size, and the memory gap on fresh markets",
             "", "  Part A, simulations (leaky -> leak-free, memory gap):"]
    for r in res["sim"]:
        lines.append(f"  {r['cond']:12} seed {r['seed']}  " + "   ".join(
            f"{v} {r[v]['leaky']:+.4f} -> {r[v]['S']:+.4f} (pred {r[v]['S_pred']:+.4f})" for v in VARS))
    lines += ["", "  Part B, fresh markets (25 size x book-to-market portfolios, 1990-2026):"]
    for r in res["regions"]:
        lines.append(f"  {r['region']} ({r['members']} members, {r['days']} days)")
        for v in VARS:
            c = r[v]
            lines.append(f"    {v:8} leak-free {c['S']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]  memory gap {c['S_pred']:+.4f}  "
                         f"leaky {c['leaky']:+.4f}  virtual " + " ".join(f"{s:+.4f}" for s in r["virtual"][v]))
    lines += ["", "  Part C, reanalysis (e67's measured share, leaky -> leak-free):"]
    for r in res["c_sim"]:
        tag = (f"{r['kind']:9} N {r['n']:3d} p {r['p']:<6} rho_c {r['rho_c']}" if "n" in r
               else f"{r['kind']:9} {r['cond']:12} seed {r['seed']}")
        lines.append(f"  {tag:42} {r['leaky']:+.4f} -> {r['S']:+.4f}  S_pred {r['S_pred']:+.4f}")
    areas = res["areas"]
    if areas:
        s, k, p = (np.array([a[q] for a in areas]) for q in ("S", "leaky", "S_pred"))
        lines += ["", f"  Brain areas ({len(areas)}): median leaky {np.median(k):+.4f} -> leak-free {np.median(s):+.4f}, "
                      f"median S_pred {np.median(p):+.4f}; median change {np.median(s - k):+.4f}; "
                      f"Spearman(S, S_pred) leaky {e67._spearman(k, p):+.3f} -> {e67._spearman(s, p):+.3f}"]
    lines += ["", "  ALM sessions (leaky -> leak-free; virtual and beyond-drift leak-free):"]
    for r in res["alm"]:
        lines.append(f"  {r['study']:11} {r['session'][:38]:38} S {r['leaky']:+.4f} -> {r['S']:+.4f}  "
                     f"S_pred {r['S_pred']:+.4f}  virtual {r['virtual']:+.4f}  drift {r['drift']:+.4f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from experiments import finance_data as fd
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    sims = [(c, s) for c in CONDS for s in A_SEEDS]
    lin = ([("factor", n, r, rho, 0) for n in e67.N_GRID for rho in e67.RHO_C for r in e67.R_GRID]
           + [("meanfield", n, k, None, 0) for n in e67.N_GRID for k in e67.KAPPA_MF]
           + [("local", n, k, None, 0) for n in e67.N_GRID[1:] for k in e67.KAPPA_LOCAL])
    garch = [(c, s) for c in CONDS for s in (0, 1)]
    spiking = [(c, s) for c in ("independent", "coupled", "hidden") for s in (0, 1)]
    sessions, alm = list(range(39)), alm_sessions()
    if quick:
        sims, lin, garch, spiking, sessions, alm = sims[::5], lin[::9], garch[:1], spiking[:1], [0], alm[::19]
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = {"sim": list(pool.map(_sim, sims)), "regions": list(pool.map(_region, fd.REGIONS)),
               "c_sim": list(pool.map(_c_linear, lin)) + list(pool.map(_c_garch, garch))
               + list(pool.map(_c_spiking, spiking)),
               "areas": [a for out in pool.map(_c_area, sessions) for a in out], "alm": list(pool.map(_c_alm, alm))}
    (out_dir / "e76_leak_audit_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e76_figures import make_figures
    make_figures(res, out_dir / "e76_figures", text)
    return text


if __name__ == "__main__":
    main()
