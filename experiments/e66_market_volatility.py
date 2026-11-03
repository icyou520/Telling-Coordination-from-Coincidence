"""Experiment 66: a new domain. How much of market volatility is coordination between its parts?

The interaction share asks how much of a group's forecast needs the members' coordination, beyond what each member's
own history predicts. In a stock market the group is the market's volatility and the members are industries (or
stocks). Each member's volatility has its own memory (volatility clustering). The question is whether the market's
volatility is forecast better by its own history than by adding up the members' own forecasts. If members move
together through spillovers or a common driver, it is; if they are independent, it is not.

Data:
  industries  Kenneth French's 49 industry portfolios, value-weighted daily returns, 1972 - Sep 2026.
              Member variable: |return| (%).
  stocks      S&P 500 constituents with every trading day, Feb 2013 - Feb 2018 (replication).
              Member variable: daily log range 100 log(high/low), a less noisy volatility measure.
Group: phi_t = mean over members of the member variable.
Twin: per member, linear ridge (penalty 1, standardized) on its own HAR history: yesterday, mean of the last 5 days,
mean of the last 22 days, and day-of-week dummies (recorded causes). Group law: the same on phi's HAR history.
S_proj = 1 - MSE(law) / MSE(law's regression fitted to the sum of twin forecasts) (e62's projected share).
Every model is cross-fitted: days are split into 250-day blocks, assigned to 3 folds in turn; each block is
forecast by models fitted on the other folds. 90% interval: bootstrap over blocks (400 resamples).
Virtual groups: each member's series circularly shifted by its own random offset (>= 500 days), 5 draws. This keeps
every member's dynamics and destroys their alignment.

Part A, simulated markets (49 members, 14000 days, 5 seeds), GARCH(1,1) members (alpha 0.08, beta 0.9):
  independent   no link;
  correlated    shocks correlated 0.5 across members, volatilities independent (co-movement without coordination
                of volatility);
  spillover     half of each member's volatility response is to yesterday's mean squared return of all members;
  factor        returns = 0.8 x a common GARCH factor + 0.6 x member GARCH noise (an unrecorded common driver).

Development on simulations only (seeds 100-102; no real share computed): independent and correlated markets gave
about -0.002, spillover +0.027 to +0.035, and a common factor about 0 at loadings 0.6, 0.8 and 0.9. A common driver
that is visible in every member's own history is learned by every twin, so it is not counted as coordination; only
links that a member's own history does not reveal are (here, responding to other members' shocks). This differs from
neurons (e62), where noisy spike counts reveal a shared input poorly and it was flagged.

Predictions, fixed before running:
  F1  Independent: |S| <= 0.02 in >= 4/5 seeds.
  F2  Correlated shocks without volatility links are not coordination: |S| <= 0.02 in >= 4/5 seeds.
  F3  Spillover is detected: S >= 0.015 and 90% lower bound > 0 in >= 4/5 seeds.
  F4  A common driver visible in every member's history is not counted: |S| <= 0.02 in >= 4/5 seeds.
  F5  Virtual groups are clean in simulations: |S_virtual| <= 0.02 in >= 90% of cells.
  R1  Virtual groups are clean on real data: |S_virtual| <= 0.02 in all draws, industries and stocks.
  R2  Market volatility is shaped by coordination: industries S >= 0.015, 90% lower bound > 0.
  R3  Replication on stocks: S >= 0.015, 90% lower bound > 0.
  R4  Coordination rises in turbulent times: Spearman correlation across calendar years between mean phi and the
      year's S (industries, out-of-sample errors of that year) >= 0.3.
  R5  Stable over time: S >= 0.015 in both 1972-1997 and 1998-2026 (industries).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 66
LAM = 1.0
BLOCK = 250
LAGS = 22
N_BOOT = 400
VIRTUAL = 5
SIM_SEEDS = tuple(range(5))
SIM_CONDITIONS = ("independent", "correlated", "spillover", "factor")
SIM_N, SIM_T = 49, 14000
SPILL = 0.5


def har(x):
    """HAR features of series x (days, ...) for days LAGS..end: yesterday, last-5 mean, last-22 mean."""
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
    t = np.arange(LAGS, len(x))
    return np.stack([x[t - 1], (c[t] - c[t - 5]) / 5, (c[t] - c[t - 22]) / 22], -1)


def _ridge(x, y, train, lam=LAM):
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-12
    z = (x - mu) / sd
    z1 = np.column_stack([np.ones(len(z)), z])
    a = z1[train].T @ z1[train] + lam * np.diag([0.0] + [1.0] * z.shape[1])
    return z1 @ np.linalg.solve(a, z1[train].T @ y[train])


def folds(n):
    block = np.arange(n) // BLOCK
    return block % 3, block


def share(x, dow):
    """Cross-fitted projected share. x (days, members); dow (days,) day-of-week 0-4. Returns per-day errors."""
    days, n = x.shape
    env = np.eye(5)[dow[LAGS:]][:, 1:]
    fold, block = folds(days - LAGS)
    feats = har(x)
    y = x[LAGS:]
    pred = np.empty_like(y)
    for i in range(n):
        xi = np.column_stack([feats[:, i], env])
        for f in range(3):
            pred[fold == f, i] = _ridge(xi, y[:, i], fold != f)[fold == f]
    phi = x.mean(1)
    xg = np.column_stack([har(phi), env])
    yg, y0 = phi[LAGS:], pred.mean(1)
    e_g, e_0 = np.empty(len(yg)), np.empty(len(yg))
    for f in range(3):
        te = fold == f
        e_g[te] = (yg[te] - _ridge(xg, yg, ~te)[te]) ** 2
        e_0[te] = (yg[te] - _ridge(xg, y0, ~te)[te]) ** 2
    return e_g, e_0, block


def summarize(e_g, e_0, block, rng):
    s = 1 - e_g.sum() / e_0.sum()
    nb = block.max() + 1
    bg, b0 = np.bincount(block, e_g, nb), np.bincount(block, e_0, nb)
    idx = rng.integers(0, nb, (N_BOOT, nb))
    boot = 1 - bg[idx].sum(1) / b0[idx].sum(1)
    lo, hi = np.quantile(boot, [0.05, 0.95])
    return {"S": float(s), "lo": float(lo), "hi": float(hi)}


def virtual(x, rng):
    days = len(x)
    return np.stack([np.roll(x[:, i], rng.integers(500, days - 500)) for i in range(x.shape[1])], 1)


def simulate(cond, seed, n=SIM_N, t=SIM_T):
    """Daily returns (%) of n GARCH(1,1) members; returns |r|."""
    rng = np.random.default_rng([SEED, seed, SIM_CONDITIONS.index(cond)])
    a, b = 0.08, 0.9
    w = 1.0 * (1 - a - b)
    if cond == "correlated":
        z = np.sqrt(0.5) * rng.standard_normal((t, 1)) + np.sqrt(0.5) * rng.standard_normal((t, n))
    else:
        z = rng.standard_normal((t, n))
    h = np.ones(n)
    r = np.zeros((t, n))
    hf, f = 1.0, np.zeros(t)
    zf = rng.standard_normal(t)
    for k in range(t):
        if cond == "factor":
            f[k] = np.sqrt(hf) * zf[k]
            hf = w + a * f[k] ** 2 + b * hf
            r[k] = 0.8 * f[k] + 0.6 * np.sqrt(h) * z[k]
            h = w + a * h * z[k] ** 2 + b * h
            continue
        r[k] = np.sqrt(h) * z[k]
        shock = r[k] ** 2
        if cond == "spillover":
            shock = (1 - SPILL) * shock + SPILL * shock.mean()
        h = w + a * shock + b * h
    return np.abs(r)


def _dow(days):
    return np.arange(days) % 5


def _dow_dates(dates):
    import datetime as dt
    return np.array([dt.date(d // 10000, d // 100 % 100, d % 100).weekday() for d in dates]).clip(0, 4)


def _sim_run(job):
    cond, seed = job
    x = simulate(cond, seed)
    rng = np.random.default_rng([SEED, 1, seed, SIM_CONDITIONS.index(cond)])
    dow = _dow(len(x))
    out = summarize(*share(x, dow), rng)
    out["virtual"] = [summarize(*share(virtual(x, rng), dow), rng)["S"] for _ in range(2)]
    return {"cond": cond, "seed": seed, **out}


def _real_run(name):
    from experiments import finance_data as fd
    dates, x, names = fd.industries() if name == "industries" else fd.stocks()
    if name == "industries":
        x = np.abs(x)
    rng = np.random.default_rng([SEED, 2, len(name)])
    dow = _dow_dates(dates)
    e_g, e_0, block = share(x, dow)
    out = {"name": name, "members": x.shape[1], "days": len(x), **summarize(e_g, e_0, block, rng)}
    out["virtual"] = [summarize(*share(virtual(x, rng), dow), rng)["S"] for _ in range(VIRTUAL)]
    d = dates[LAGS:]
    years = d // 10000
    phi = x.mean(1)[LAGS:]
    out["years"] = {int(y): {"S": float(1 - e_g[years == y].sum() / e_0[years == y].sum()),
                             "phi": float(phi[years == y].mean())} for y in np.unique(years)}
    for lab, sel in (("early", d < 19980101), ("late", d >= 19980101)):
        out[lab] = float(1 - e_g[sel].sum() / e_0[sel].sum())
    return out


def predictions(res):
    sim = {c: [r for r in res["sim"] if r["cond"] == c] for c in SIM_CONDITIONS}
    ns = len(SIM_SEEDS)
    f1 = sum(abs(r["S"]) <= 0.02 for r in sim["independent"])
    f2 = sum(abs(r["S"]) <= 0.02 for r in sim["correlated"])
    f3 = sum(r["S"] >= 0.015 and r["lo"] > 0 for r in sim["spillover"])
    f4 = sum(abs(r["S"]) <= 0.02 for r in sim["factor"])
    cells = [v for r in res["sim"] for v in r["virtual"]]
    f5 = sum(abs(v) <= 0.02 for v in cells)
    real = {r["name"]: r for r in res["real"]}
    rv = [v for r in res["real"] for v in r["virtual"]]
    r1 = sum(abs(v) <= 0.02 for v in rv)
    ind, stk = real.get("industries"), real.get("stocks")
    from scipy.stats import spearmanr
    rho = (float(spearmanr([v["phi"] for v in ind["years"].values()], [v["S"] for v in ind["years"].values()])[0])
           if ind else np.nan)
    fmt = lambda r: f"S {r['S']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]" if r else "n/a"  # noqa: E731
    return [
        ("F1", "simulated independent: |S| <= 0.02, >= 4/5 seeds", f1 >= 4, f"{f1}/{ns}"),
        ("F2", "simulated correlated shocks, no volatility links: |S| <= 0.02, >= 4/5", f2 >= 4, f"{f2}/{ns}"),
        ("F3", "simulated spillover detected: S >= 0.015, lower bound > 0, >= 4/5", f3 >= 4, f"{f3}/{ns}"),
        ("F4", "simulated common driver visible in own histories not counted: |S| <= 0.02, >= 4/5", f4 >= 4,
         f"{f4}/{ns}"),
        ("F5", "virtual groups clean in simulations: |S_virtual| <= 0.02 in >= 90%", f5 >= 0.9 * len(cells),
         f"{f5}/{len(cells)}"),
        ("R1", "virtual groups clean on real data: all |S_virtual| <= 0.02", bool(rv) and r1 == len(rv),
         f"{r1}/{len(rv)}"),
        ("R2", "industries: S >= 0.015, 90% lower bound > 0", bool(ind) and ind["S"] >= 0.015 and ind["lo"] > 0,
         fmt(ind)),
        ("R3", "stocks replication: S >= 0.015, 90% lower bound > 0",
         bool(stk) and stk["S"] >= 0.015 and stk["lo"] > 0, fmt(stk)),
        ("R4", "coordination rises in turbulent years: Spearman(mean phi, S) >= 0.3", rho >= 0.3, f"rho {rho:+.2f}"),
        ("R5", "stable: S >= 0.015 in 1972-1997 and 1998-2026",
         bool(ind) and min(ind["early"], ind["late"]) >= 0.015,
         f"{ind['early']:+.3f} / {ind['late']:+.3f}" if ind else "n/a"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 66: how much of market volatility is coordination between its parts?", "",
             "  Simulated markets (S [90%], virtual):"]
    for r in res["sim"]:
        lines.append(f"  {r['cond']:12} seed {r['seed']}  S {r['S']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                     f"virtual {' '.join(f'{v:+.3f}' for v in r['virtual'])}")
    lines += ["", "  Real markets:"]
    for r in res["real"]:
        lines.append(f"  {r['name']:11} members {r['members']}  days {r['days']}  S {r['S']:+.3f} "
                     f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  virtual {' '.join(f'{v:+.3f}' for v in r['virtual'])}")
        if r["name"] == "industries":
            lines.append(f"  {'':11} 1972-1997 {r['early']:+.3f}  1998-2026 {r['late']:+.3f}")
            lines.append("  " + "  ".join(f"{y}:{v['S']:+.2f}" for y, v in r["years"].items()))
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from concurrent.futures import ProcessPoolExecutor
    out_dir = out_dir or Path("results")
    jobs = [(c, s) for c in SIM_CONDITIONS for s in SIM_SEEDS]
    if quick:
        jobs = jobs[::5]
    with ProcessPoolExecutor(max_workers=6) as pool:
        sim = list(pool.map(_sim_run, jobs))
        real = list(pool.map(_real_run, ["industries", "stocks"]))
    res = {"sim": sim, "real": real}
    (out_dir / "e66_market_volatility_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e66_figures import make_figures
    make_figures(res, out_dir / "e66_figures", text)
    return text


if __name__ == "__main__":
    main()
