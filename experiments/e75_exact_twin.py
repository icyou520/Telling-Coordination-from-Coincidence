"""Experiment 75: an exact memory twin for non-additive group variables (no virtual-floor correction).

e74 extended the memory gap to any group variable. Sums of a member transform reduce exactly; for non-additive
variables (the worst industry) a Monte Carlo twin (each member's HAR forecast plus its own residuals drawn by forecast
decile) was approximate: independent markets and virtual groups scored about 0.015, so shares had to be read
against a virtual floor. Development here (e66 seeds 100-105 only) found why and a fix.

  Why. Under independence the exact twin is E[G | every member's own history], and then the share is 0 by the tower
  property. An oracle twin that knows each simulated member's true GARCH volatility scored -0.001 to +0.002. e74's
  twin conditions each member on its forecast level only; the group's history (yesterday's worst industry) still
  reveals members' recent shocks beyond that level, which the law then uses. Adding the member's own recent maxima
  and squared levels halved the bias (0.000-0.015), but did not remove it.
  The fix, a volatility twin. Each member is modelled where volatility lives, in returns: its scale sigma is a
  cross-fitted ridge of its |return| on exponentially weighted means of its own past |returns| (decays 0.8, 0.9,
  0.97, plus day-of-week), floored at 0.25 x its median |return|; its standardized |return| u = |r| / sigma is drawn
  from its own out-of-fold days; log volatility is rebuilt as log(sigma u + 0.05). A first version on squared
  returns was exact on long series but in 2,800-day series some fitted variances collapsed and virtual groups hit
  +0.26; the |return| version with the floor: independent markets |S| <= 0.003 (14,000 days), virtual groups within
  +-0.011 except one -0.017 (2,800 days, 37 members), with no floor subtracted.
  Joint twin (as e74): every member's u from the same training day, keeping same-day dependence.

Group variables (members: industries' log volatility, log(|r| + 0.05)):
  mean, breadth (fraction above their own 95th percentile): additive, linear twin and memory gap as e74;
  max (the worst industry) and systemic (1 if at least 20% of industries are above their own 95th percentile, a
  threshold of a sum; under independence such days essentially never occur): the volatility twin, independent and
  joint. Same-day part = S - S_joint.
  For comparison, e74's twin on the max (raw S, no floor).

Development findings that shape the predictions (seeds 100-105, 120-122): correlated shocks gave max 0.18-0.20 and
systemic 0.06-0.07, both about 0 with the joint twin (one -0.015); a common factor gave max 0.33-0.37 and systemic
0.05-0.07, with joint twins 0.003-0.010 (the squared-return version had suggested 0.013-0.043, a dynamic part from the
factor's changing volatility; with the robust twin it is small). Spillover: max 0.005-0.019, systemic about 0. In
2,800-day series the cross-fitted shares of mean and breadth ran 0.003-0.020 below their memory-gap predictions
(e71 saw the same short-sample bias), so R4's tolerance is 0.02.

Data. Simulations: e66's markets, fresh seeds 20-24. Real, primary: value-weighted industry returns 1926-1935
(37 industries with no missing day, 2,824 days, never analysed; includes 1929). Replication: equal-weighted
1926-1935. Secondary: e74's three sets, compared with e74's floor-corrected shares. Pipeline otherwise e66/e74
(250-day blocks in 3 folds, 400 bootstrap resamples, 5 virtual groups by circular shifts >= 500 days).

Predictions, fixed before running:
  X1  Exact under independence: |S| <= 0.005 for max and systemic, nothing subtracted, >= 4/5 seeds each.
  X2  Virtual groups in simulations: |S| <= 0.01 for max and systemic in >= 95% of draws.
  X3  Same-day dependence caught and isolated: correlated shocks and common factor, max S >= 0.1 and systemic
      S >= 0.03 (5/5 each); correlated shocks, |S_joint| <= 0.01 for max and systemic (>= 4/5 each).
  X4  A common factor is mostly same-day: factor, same-day part of the max >= 0.25 in 5/5.
  X5  e74's twin was biased: its raw max S in independent markets >= 0.008 in >= 4/5.
  R1  Real virtual groups, primary and replication: all |S| <= 0.02 for max and systemic, nothing subtracted.
  R2  The worst industry is mostly same-day co-movement: same-day part >= 0.1, primary and replication.
  R3  Systemic days are coordination: S >= 0.03 with 90% lower bound > 0, primary and replication.
  R4  Breadth reduces to the memory gap on fresh data: |S - S_pred| <= 0.02, primary and replication.
  R5  e74's floor correction was adequate: on e74's three real sets, |S_max - e74 excess| <= 0.05.

Correction after the first run (predictions unchanged): e74's set named "primary" overwrote this experiment's
primary in REAL, so the 1926-1935 value-weighted set was never run and R1-R4 were scored on e74's primary. e74's sets
are now prefixed "e74 ". The first run's output is kept in results/e75_exact_twin_run1*.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments import e66_market_volatility as e66
from experiments import e74_nonlinear_gap as e74

SEED = 75
DRAWS = 200
VIRTUAL = 5
CONDS = e74.CONDS
SIM_SEEDS = tuple(range(20, 25))
DECAYS = (0.8, 0.9, 0.97)
FLOOR = 0.25
OFFSET = e74.OFFSET
TAIL = 0.95
SYSTEMIC = 0.2
NONADD = ("max", "systemic")
REAL = {"primary": ("Average Value Weighted Returns", 19260101, 19351231),
        "replication": ("Average Equal Weighted Returns", 19260101, 19351231),
        **{f"e74 {k}": v for k, v in e74.REAL.items()}}


def ewma(x, lam):
    """Exponentially weighted mean of x up to yesterday, for days LAGS..end."""
    out = np.empty_like(x)
    s = x[:e66.LAGS].mean(0)
    for k in range(len(x)):
        out[k] = s
        s = lam * s + (1 - lam) * x[k]
    return out[e66.LAGS:]


def volatility(l, dow):
    """Each member's volatility from its own past squared returns (cross-fitted ridge on r^2), and its out-of-fold
    standardized |return| u. The twin rebuilds log volatility as log(sigma u + offset)."""
    a = np.exp(l) - OFFSET
    env = np.eye(5)[dow[e66.LAGS:]][:, 1:]
    fold, block = e66.folds(len(l) - e66.LAGS)
    feats = np.stack([ewma(a, d) for d in DECAYS], -1)
    y = a[e66.LAGS:]
    sig = np.empty_like(y)
    for i in range(l.shape[1]):
        xi = np.column_stack([feats[:, i], env])
        for f in range(3):
            tr, te = fold != f, fold == f
            sig[te, i] = np.maximum(e66._ridge(xi, y[:, i], tr)[te], FLOOR * np.median(y[tr, i]))
    return sig, y / sig, fold, block, env


def vol_twin(sig, u, fold, rng, stat, joint=False, chunk=400):
    """E[stat(log(sigma_i u_i + offset))], u_i from member i's own out-of-fold standardized |returns|, independent
    across members or (joint) all from the same training day."""
    out = np.empty(len(sig))
    n = sig.shape[1]
    for f in range(3):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        days = rng.integers(0, len(tr), DRAWS if joint else (DRAWS, n))
        uu = u[tr][days] if joint else u[tr][days, np.arange(n)]
        for k in range(0, len(te), chunk):
            t = te[k:k + chunk]
            out[t] = stat(np.log(sig[t][:, None, :] * uu[None] + OFFSET)).mean(1)
    return out


def shares(l, dow, rng, full=True):
    """S for every group variable. full=False: only the volatility twin's independent shares (for virtual groups)."""
    c = np.quantile(l, TAIL, 0)
    stats = {"max": lambda v: v.max(-1), "systemic": lambda v: ((v > c).mean(-1) >= SYSTEMIC).astype(float)}
    groups = {"max": l.max(1), "systemic": ((l > c).mean(1) >= SYSTEMIC).astype(float)}
    sig, u, fold, block, env = volatility(l, dow)
    out = {}
    for v in NONADD:
        if groups[v][e66.LAGS:].std() == 0:
            out[v] = {"S": 0.0, "lo": 0.0, "hi": 0.0, "joint": 0.0, "same_day": 0.0}
            continue
        e_g, e_0 = e74.law_errors(groups[v], vol_twin(sig, u, fold, rng, stats[v]), env, fold)
        out[v] = e66.summarize(e_g, e_0, block, rng)
        if full:
            e_g, e_0 = e74.law_errors(groups[v], vol_twin(sig, u, fold, rng, stats[v], joint=True), env, fold)
            out[v]["joint"] = float(1 - e_g.sum() / e_0.sum())
            out[v]["same_day"] = out[v]["S"] - out[v]["joint"]
    out["systemic"]["rate"] = float(groups["systemic"].mean())
    if full:
        add = e74.shares(l, dow, rng)
        for v in ("mean", "breadth"):
            out[v] = add[v]
        out["max"]["e74"] = add["max"]["S"]
    return out


def run_series(l, dow, rng):
    out = shares(l, dow, rng)
    out["virtual"] = {v: [] for v in NONADD}
    for _ in range(VIRTUAL):
        s = shares(e66.virtual(l, rng), dow, rng, full=False)
        for v in NONADD:
            out["virtual"][v].append(s[v]["S"])
    return out


def _sim(job):
    cond, seed = job
    l = np.log(e66.simulate(cond, seed) + OFFSET)
    rng = np.random.default_rng([SEED, seed, CONDS.index(cond)])
    return {"cond": cond, "seed": seed, **run_series(l, e66._dow(len(l)), rng)}


def _real(name):
    from experiments import finance_data as fd
    title, start, end = REAL[name]
    dates, r, names = fd.industries(title, start, end)
    rng = np.random.default_rng([SEED, len(name)])
    return {"name": name, "members": len(names), "days": len(dates),
            **run_series(np.log(np.abs(r) + OFFSET), e66._dow_dates(dates), rng)}


def predictions(res, e74_res=None):
    sim = {c: [r for r in res["sim"] if r["cond"] == c] for c in CONDS}
    ns = len(SIM_SEEDS)

    def cnt(c, ok):
        return sum(bool(ok(r)) for r in sim[c])

    x1 = {v: cnt("independent", lambda r: abs(r[v]["S"]) <= 0.005) for v in NONADD}
    virt = [s for r in res["sim"] for v in NONADD for s in r["virtual"][v]]
    x3 = {f"{c} {v}": cnt(c, lambda r: r[v]["S"] >= (0.1 if v == "max" else 0.03))
          for c in ("correlated", "factor") for v in NONADD}
    x3j = {f"joint {v}": cnt("correlated", lambda r: abs(r[v]["joint"]) <= 0.01) for v in NONADD}
    x4 = cnt("factor", lambda r: r["max"]["same_day"] >= 0.25)
    x5 = cnt("independent", lambda r: r["max"]["e74"] >= 0.008)
    fmt = lambda d: ", ".join(f"{k} {v}/{ns}" for k, v in d.items())  # noqa: E731
    out = [
        ("X1", "exact under independence: |S| <= 0.005, max and systemic, nothing subtracted, >= 4/5",
         min(x1.values()) >= 4, fmt(x1)),
        ("X2", "simulated virtual groups: |S| <= 0.01 in >= 95%", sum(abs(s) <= 0.01 for s in virt) >= 0.95 * len(virt),
         f"{sum(abs(s) <= 0.01 for s in virt)}/{len(virt)}, largest {max(abs(s) for s in virt):.4f}"),
        ("X3", "same-day dependence caught (max >= 0.1, systemic >= 0.03, 5/5) and isolated (joint ~ 0, >= 4/5)",
         min(x3.values()) == ns and min(x3j.values()) >= 4, fmt({**x3, **x3j})),
        ("X4", "a common factor is mostly same-day: same-day part of the max >= 0.25, 5/5", x4 == ns, f"{x4}/{ns}"),
        ("X5", "e74's twin was biased: raw max S >= 0.008 in independent markets, >= 4/5", x5 >= 4, f"{x5}/{ns}"),
    ]
    real = {r["name"]: r for r in res["real"]}
    pr = [real[k] for k in ("primary", "replication") if k in real]
    if not pr:
        return out
    rv = [s for r in pr for v in NONADD for s in r["virtual"][v]]
    out += [
        ("R1", "real virtual groups: all |S| <= 0.02, max and systemic, nothing subtracted",
         all(abs(s) <= 0.02 for s in rv), f"largest {max(abs(s) for s in rv):.4f} of {len(rv)}"),
        ("R2", "worst industry mostly same-day: same-day part >= 0.1, primary and replication",
         all(r["max"]["same_day"] >= 0.1 for r in pr), ", ".join(f"{r['name']} {r['max']['same_day']:+.3f}" for r in pr)),
        ("R3", "systemic days are coordination: S >= 0.03, lower bound > 0, primary and replication",
         all(r["systemic"]["S"] >= 0.03 and r["systemic"]["lo"] > 0 for r in pr),
         ", ".join(f"{r['name']} {r['systemic']['S']:+.3f} [{r['systemic']['lo']:+.3f}]" for r in pr)),
        ("R4", "breadth reduces to the memory gap on fresh data: |S - S_pred| <= 0.02, primary and replication",
         all(abs(r["breadth"]["S"] - r["breadth"]["S_pred"]) <= 0.02 for r in pr),
         ", ".join(f"{r['name']} {r['breadth']['S']:+.4f} vs {r['breadth']['S_pred']:+.4f}" for r in pr)),
    ]
    if e74_res:
        old = {f"e74 {r['name']}": r["max"]["excess"] for r in e74_res["real"]}
        pairs = [(k, real[k]["max"]["S"], old[k]) for k in old if k in real]
        out.append(("R5", "e74's floor correction was adequate: |S_max - e74 excess| <= 0.05 on its three sets",
                    bool(pairs) and all(abs(a - b) <= 0.05 for _, a, b in pairs),
                    ", ".join(f"{k} {a:+.3f} vs {b:+.3f}" for k, a, b in pairs)))
    return out


def _e74():
    p = Path("results/e74_nonlinear_gap_raw.json")
    return json.loads(p.read_text()) if p.exists() else None


def analyze(res, e74_res=None) -> str:
    lines = ["=" * 78, "Experiment 75: an exact memory twin for non-additive group variables", "",
             "  S (nothing subtracted); joint twin; same-day part; virtual groups:"]

    def row(label, r):
        s = f"    {label:22}"
        for v in NONADD:
            s += (f"  {v} {r[v]['S']:+.4f} joint {r[v]['joint']:+.4f} virt "
                  f"[{min(r['virtual'][v]):+.4f}, {max(r['virtual'][v]):+.4f}]")
        return s + f"  e74 max {r['max']['e74']:+.4f}  systemic rate {r['systemic']['rate']:.3f}"

    for r in res["sim"]:
        lines.append(row(f"{r['cond']} {r['seed']}", r))
    lines += ["", "  Real markets:"]
    for r in res["real"]:
        lines.append(row(f"{r['name']} ({r['members']}, {r['days']}d)", r))
        lines.append(f"    {'':22}  90%: max [{r['max']['lo']:+.4f}, {r['max']['hi']:+.4f}]  systemic "
                     f"[{r['systemic']['lo']:+.4f}, {r['systemic']['hi']:+.4f}]  mean {r['mean']['S']:+.4f} "
                     f"(pred {r['mean']['S_pred']:+.4f})  breadth {r['breadth']['S']:+.4f} (pred {r['breadth']['S_pred']:+.4f})")
    lines += ["", "  Pre-registered predictions:"]
    for k, desc, ok, detail in predictions(res, e74_res):
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
    (out_dir / "e75_exact_twin_raw.json").write_text(json.dumps(res, default=float))
    e74_res = _e74()
    text = analyze(res, e74_res)
    from experiments.e75_figures import make_figures
    make_figures(res, out_dir / "e75_figures", text, e74_res)
    return text


if __name__ == "__main__":
    main()
