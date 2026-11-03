"""Experiment 74: the memory gap for nonlinear group variables (market breadth and the worst industry).

The memory-gap equation (e67) predicts the interaction share of a group variable that is a sum of its members. e72
extended it to polarization (quadratic). Here the general case, on markets (log volatility of industries, e69).

  Additive variables, G = sum_i h(x_i) for any member transform h, reduce exactly to the linear memory gap on the
  transformed members h(x_i). Tested on breadth: the share of industries above their own 95th percentile of log
  volatility (h = indicator), next to the mean log volatility (h = identity).
  Non-additive variables (here the worst industry's log volatility, the max) also depend on how members move
  together on the same day, which no member's history shows. Their twin is a Monte Carlo memory twin: each
  member's own HAR forecast plus a draw of its own residuals (on other folds, on days when its forecast was in the
  same decile, because the spread grows with the forecast), independent across members; G0 = E[max] over 200 draws.
  A joint twin draws every member at the rank its residual had on the same past day (the empirical copula of
  residuals), keeping same-day dependence: its share is the dynamic coordination only; the difference between the
  two is the same-day part.
  The Monte Carlo twin is not exact (in development, independent simulated markets gave max shares 0.014-0.027,
  and so did their virtual groups), so every share is reported as excess over its virtual floor: the median over
  5 virtual groups (members circularly shifted by >= 500 days) of the same measurement.

Pipeline otherwise e66/e69: cross-fitted HAR twin and law (day-of-week dummies), 250-day blocks in 3 folds, 400
block-bootstrap resamples; members log(|return| + 0.05).

Developed on e66's simulated markets, development seeds 100-102 only (no real data touched): mean and breadth
shares matched their memory-gap predictions within 0.009 (mostly 0.003); excess shares of the max: independent
-0.003 to 0.006, correlated shocks 0.18-0.20 (joint -0.005 to 0.002), spillover 0.015-0.022, common factor 0.30-0.36
(joint -0.011 to -0.003). Confirmation uses fresh seeds 10-14.

Real data: primary, equal-weighted industry returns 1936-1971 (the industries with no missing day; never analysed).
Secondary, without predictions: value-weighted 1936-1971 and equal-weighted 1972-2026 (their mean log volatility
was e69's; breadth and max never computed).

Predictions, fixed before running ("excess" = S minus its virtual floor):
  N1  Additive reduction: |S - S_pred| <= 0.01 for mean and breadth in >= 90% of simulated runs.
  N2  The Monte Carlo twin agrees with the linear twin on the mean: |S_mc - S| <= 0.01 in >= 90% of runs.
  N3  Independent markets: |excess| <= 0.01 for mean, breadth, max and joint max in >= 4/5 seeds each.
  N4  Correlated shocks: max excess >= 0.1 in 5/5; |excess| <= 0.01 for mean and breadth and <= 0.015 for joint
      max in >= 4/5 each.
  N5  Common factor: max excess >= 0.1 in 5/5; |joint max excess| <= 0.015 in >= 4/5.
  N6  Spillover reaches the max: max excess >= 0.01 in >= 4/5.
  R1  Real virtual groups clean for additive variables: all 5 virtual |S| <= 0.01, mean and breadth.
  R2  Breadth reduces to the memory gap on real data: |S - S_pred| <= 0.01, and S >= 0.005 with lower bound > 0.
  R3  Same-day co-movement dominates the worst industry: max excess >= 0.1.
  R4  Contagion of extremes: dynamic coordination in the worst industry exceeds that in the average, joint max
      excess >= mean excess + 0.005 (a hypothesis; the alternative is that extremes carry no extra coordination).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments import e66_market_volatility as e66
from experiments.e67_memory_gap import HAR_WINDOWS, autocov, s_pred, series_design

SEED = 74
OFFSET = 0.05
TAIL = 0.95
DRAWS = 200
BINS = 10
VIRTUAL = 5
CONDS = ("independent", "correlated", "spillover", "factor")
SIM_SEEDS = tuple(range(10, 15))
VARS = ("mean", "breadth", "max", "max_joint")


def members_forecast(x, dow):
    """Cross-fitted HAR forecasts of every member from its own history (e66 twin). Returns forecasts and the
    out-of-fold residuals (days - LAGS, members), fold labels and blocks."""
    env = np.eye(5)[dow[e66.LAGS:]][:, 1:]
    fold, block = e66.folds(len(x) - e66.LAGS)
    feats, y = e66.har(x), x[e66.LAGS:]
    pred = np.empty_like(y)
    for i in range(x.shape[1]):
        xi = np.column_stack([feats[:, i], env])
        for f in range(3):
            pred[fold == f, i] = e66._ridge(xi, y[:, i], fold != f)[fold == f]
    return pred, y - pred, fold, block, env


def law_errors(g, g0, env, fold):
    """Group law on the group variable's own HAR history; its projection fitted to the twin forecast g0."""
    xg = np.column_stack([e66.har(g), env])
    yg = g[e66.LAGS:]
    e_g, e_0 = np.empty(len(yg)), np.empty(len(yg))
    for f in range(3):
        te = fold == f
        e_g[te] = (yg[te] - e66._ridge(xg, yg, ~te)[te]) ** 2
        e_0[te] = (yg[te] - e66._ridge(xg, g0, ~te)[te]) ** 2
    return e_g, e_0


def memory_twin(pred, resid, fold, rng, stat=np.max, joint=False, chunk=500):
    """E[stat_i (forecast_i + e_i)], each e_i drawn from member i's own residuals on other folds on days when its
    forecast was in the same decile (residual spread grows with the forecast). Independent across members (the
    memory twin of a non-additive group variable), or joint: every member's draw at the rank its residual had on
    the same training day, keeping same-day dependence (empirical copula) and removing only dynamic coordination."""
    out = np.empty(len(pred))
    n = pred.shape[1]
    for f in range(3):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        edges = np.quantile(pred[tr], np.linspace(0, 1, BINS + 1)[1:-1], axis=0)
        rank = (resid[tr].argsort(0).argsort(0) + 0.5) / len(tr)
        days = rng.integers(0, len(tr), DRAWS if joint else (n, DRAWS))
        u = rank[days].T if joint else rank[days, np.arange(n)[:, None]]
        draws = np.empty((n, BINS, DRAWS))
        for i in range(n):
            b = np.searchsorted(edges[:, i], pred[tr, i])
            for d in range(BINS):
                draws[i, d] = np.quantile(resid[tr[b == d], i], u[i])
        for k in range(0, len(te), chunk):
            t = te[k:k + chunk]
            b = np.stack([np.searchsorted(edges[:, i], pred[t, i]) for i in range(n)], 1)
            e = draws[np.arange(n)[None], b]
            out[t] = stat(pred[t][:, :, None] + e, axis=1).mean(1)
    return out


def _pred(x):
    offs, f, y = series_design(HAR_WINDOWS)
    xc = (x - x.mean(0)).T[:, None, :]
    o, g = autocov(xc, int(offs.max() - offs.min()))
    return s_pred(o, g, offs, f, y)


def shares(l, dow, rng):
    """l (days, members) log volatility. Returns S for the mean, breadth (share of members above their own 95th
    percentile) and max, plus S_pred for the additive ones and the Monte Carlo twin's S for the mean (a check)."""
    ind = (l > np.quantile(l, TAIL, 0)).astype(float)
    out = {}
    pred, resid, fold, block, env = members_forecast(l, dow)
    for name, x in (("mean", l), ("breadth", ind)):
        if name == "mean":
            p = pred
        else:
            p = members_forecast(x, dow)[0]
        e_g, e_0 = law_errors(x.mean(1), p.mean(1), env, fold)
        out[name] = {**e66.summarize(e_g, e_0, block, rng), "S_pred": _pred(x)}
    for name, joint in (("max", False), ("max_joint", True)):
        e_g, e_0 = law_errors(l.max(1), memory_twin(pred, resid, fold, rng, joint=joint), env, fold)
        out[name] = e66.summarize(e_g, e_0, block, rng)
    e_g, e_0 = law_errors(l.mean(1), memory_twin(pred, resid, fold, rng, np.mean), env, fold)
    out["mean_mc"] = float(1 - e_g.sum() / e_0.sum())
    return out


def run_series(l, dow, rng):
    out = shares(l, dow, rng)
    out["virtual"] = {v: [] for v in VARS}
    for _ in range(VIRTUAL):
        s = shares(e66.virtual(l, rng), dow, rng)
        for v in VARS:
            out["virtual"][v].append(s[v]["S"])
    for v in VARS:
        out[v]["excess"] = out[v]["S"] - float(np.median(out["virtual"][v]))
    return out


def _sim(job):
    cond, seed = job
    l = np.log(e66.simulate(cond, seed) + OFFSET)
    rng = np.random.default_rng([SEED, seed, CONDS.index(cond)])
    return {"cond": cond, "seed": seed, **run_series(l, e66._dow(len(l)), rng)}


REAL = {"primary": ("Average Equal Weighted Returns", 19360101, 19711231),
        "value 1936-1971": ("Average Value Weighted Returns", 19360101, 19711231),
        "equal 1972-2026": ("Average Equal Weighted Returns", 19720101, 99999999)}


def _real(name):
    from experiments import finance_data as fd
    title, start, end = REAL[name]
    dates, r, names = fd.industries(title, start, end)
    rng = np.random.default_rng([SEED, len(name)])
    return {"name": name, "members": len(names), "days": len(dates),
            **run_series(np.log(np.abs(r) + OFFSET), e66._dow_dates(dates), rng)}


def predictions(res):
    sim = {c: [r for r in res["sim"] if r["cond"] == c] for c in CONDS}
    runs = res["sim"]
    gap = [abs(r[v]["S"] - r[v]["S_pred"]) for r in runs for v in ("mean", "breadth")]
    mc = [abs(r["mean_mc"] - r["mean"]["S"]) for r in runs]

    def count(c, v, ok):
        return sum(bool(ok(r[v]["excess"])) for r in sim[c])

    small = lambda e: abs(e) <= 0.01  # noqa: E731
    ns = len(SIM_SEEDS)
    n3 = {v: count("independent", v, small) for v in VARS}
    n4 = {"max": count("correlated", "max", lambda e: e >= 0.1), "mean": count("correlated", "mean", small),
          "breadth": count("correlated", "breadth", small),
          "max_joint": count("correlated", "max_joint", lambda e: abs(e) <= 0.015)}
    n5 = {"max": count("factor", "max", lambda e: e >= 0.1),
          "max_joint": count("factor", "max_joint", lambda e: abs(e) <= 0.015)}
    n6 = count("spillover", "max", lambda e: e >= 0.01)
    p = next((r for r in res["real"] if r["name"] == "primary"), None)
    fmt = lambda d: ", ".join(f"{k} {v}/{ns}" for k, v in d.items())  # noqa: E731
    out = [
        ("N1", "additive reduction: |S - S_pred| <= 0.01, mean and breadth, >= 90% of runs",
         sum(g <= 0.01 for g in gap) >= 0.9 * len(gap), f"{sum(g <= 0.01 for g in gap)}/{len(gap)}, max {max(gap):.4f}"),
        ("N2", "Monte Carlo twin agrees with the linear twin on the mean: |S_mc - S| <= 0.01, >= 90%",
         sum(g <= 0.01 for g in mc) >= 0.9 * len(mc), f"{sum(g <= 0.01 for g in mc)}/{len(mc)}, max {max(mc):.4f}"),
        ("N3", "independent: |excess| <= 0.01 for every variable, >= 4/5 each", min(n3.values()) >= 4, fmt(n3)),
        ("N4", "correlated shocks: max excess >= 0.1 (5/5); mean, breadth, joint max ~ 0 (>= 4/5)",
         n4["max"] == ns and min(n4["mean"], n4["breadth"], n4["max_joint"]) >= 4, fmt(n4)),
        ("N5", "common factor: max excess >= 0.1 (5/5); joint max ~ 0 (>= 4/5)",
         n5["max"] == ns and n5["max_joint"] >= 4, fmt(n5)),
        ("N6", "spillover reaches the max: max excess >= 0.01, >= 4/5", n6 >= 4, f"{n6}/{ns}"),
    ]
    if p is None:
        return out
    virt = [v for k in ("mean", "breadth") for v in p["virtual"][k]]
    b = p["breadth"]
    out += [
        ("R1", "real virtual groups: all |S| <= 0.01, mean and breadth", all(abs(v) <= 0.01 for v in virt),
         " ".join(f"{v:+.4f}" for v in virt)),
        ("R2", "breadth: |S - S_pred| <= 0.01, S >= 0.005, lower bound > 0",
         abs(b["S"] - b["S_pred"]) <= 0.01 and b["S"] >= 0.005 and b["lo"] > 0,
         f"S {b['S']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}], S_pred {b['S_pred']:+.4f}"),
        ("R3", "same-day co-movement dominates the worst industry: max excess >= 0.1", p["max"]["excess"] >= 0.1,
         f"max excess {p['max']['excess']:+.4f}"),
        ("R4", "contagion of extremes: joint max excess >= mean excess + 0.005",
         p["max_joint"]["excess"] >= p["mean"]["excess"] + 0.005,
         f"joint max {p['max_joint']['excess']:+.4f}, mean {p['mean']['excess']:+.4f}"),
    ]
    return out


def analyze(res) -> str:
    lines = ["=" * 78, "Experiment 74: the memory gap for nonlinear group variables (markets)", "",
             "  S (excess over virtual floor); S_pred for additive variables:"]

    def row(label, r):
        cells = []
        for v in VARS:
            c = f"{v} {r[v]['S']:+.4f} ({r[v]['excess']:+.4f})"
            if "S_pred" in r[v]:
                c += f" pred {r[v]['S_pred']:+.4f}"
            cells.append(c)
        return f"    {label:22} " + "  ".join(cells) + f"  mc-mean {r['mean_mc']:+.4f}"

    for r in res["sim"]:
        lines.append(row(f"{r['cond']} {r['seed']}", r))
    lines += ["", "  Real markets (log volatility of industries):"]
    for r in res["real"]:
        lines.append(row(f"{r['name']} ({r['members']})", r))
        lines.append(f"    {'':22} 90%: " + "  ".join(f"{v} [{r[v]['lo']:+.4f}, {r[v]['hi']:+.4f}]" for v in VARS))
        lines.append(f"    {'':22} same-day part of the max {r['max']['excess'] - r['max_joint']['excess']:+.4f}")
    lines += ["", "  Pre-registered predictions:"]
    for k, desc, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {k} {desc} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    jobs = [(c, s) for c in CONDS for s in SIM_SEEDS]
    names = list(REAL)
    if quick:
        jobs, names = jobs[::5], names[:1]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        real = ex.map(_real, names)
        sim = list(ex.map(_sim, jobs))
        real = list(real)
    res = {"sim": sim, "real": real}
    (out_dir / "e74_nonlinear_gap_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e74_figures import make_figures
    make_figures(res, out_dir / "e74_figures", text)
    return text


if __name__ == "__main__":
    main()
