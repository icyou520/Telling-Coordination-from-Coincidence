"""e79: individual stocks in eight markets. Coordination in their volatility, and the memory gap.

Every market result so far used portfolios: 49 US industries (e66-e69) and 25 size x book-to-market portfolios
in Japan, Europe and Asia Pacific (e76). A portfolio averages away most of each firm's own noise, so its
volatility history reveals the market's shared driver well. The share counts a common driver only to the extent
that members' own histories reveal it poorly (e66), so individual stocks, whose volatility is mostly their own
noise, should show more of the market's volatility as coordination. The one stock set analysed before (467 S&P
500 stocks, 2013-2018) was void: five years is too short for time-shifted virtual groups.

Data (experiments/stock_data.py): today's constituents of the S&P 100, Nikkei 225, FTSE 100, DAX, CAC 40, Hang
Seng, S&P/ASX 50 and S&P/TSX 60 (Wikipedia, fetched 2026-10-29), Yahoo Finance adjusted daily closes,
2000-01-03 to 2026-10-23 (about 6,700 days). Members are the constituents priced on >= 97% of the market's days,
at most 40 (seeded subset). Markets with fewer than 15 members are excluded. These are survivors: firms that left
since 2000 are absent.

Pipeline: e76's, unchanged. Members log(|r| + 0.05); group variables mean and breadth (share of members above
their own 95th percentile); leak-free cross-fitted share with 400 block-bootstrap resamples; memory gap from
HAR windows; 5 virtual groups (each member shifted by its own random offset). Comparators: the same pipeline on
portfolios of the same region over the same dates (US: 49 industries; Japan: Japan 25; UK, Germany, France:
Europe 25; Hong Kong, Australia: Asia Pacific ex Japan 25). Canada has none.

Development (simulations only, seeds 900-901; 400 GARCH stocks, 6,700 days; groups of 40 stocks or 25 portfolios
of 16): the memory gap matched the mean share within 0.002 and breadth within 0.012; independent markets and
virtual groups within 0.0013 of 0; with a strong common factor (load 0.8) stocks scored 0.006-0.010 against
portfolios' 0.000-0.001, with a moderate one (0.55) the two were similar (0.002-0.007 against 0.003-0.004);
volatility spillover between stocks gave stocks 0.02-0.03, portfolios 0.003.

Pre-registered predictions:
  S1 simulations (fresh seeds 910-914): |S - S_pred| <= 0.005 (mean) in >= 90% of dependent cells, stocks and
     portfolios.
  S2 simulations: independent |S| <= 0.003 (mean, stocks and portfolios) and every virtual |S| <= 0.003.
  S3 simulations: with a strong factor (0.8), stocks > portfolios in >= 4/5 seeds.
  M1 the memory gap holds for stocks: |S - S_pred| <= 0.01 (mean) in every included market.
  M2 virtual stock groups are clean: |S| <= 0.01 (mean) for >= 90% of virtual groups.
  M3 stocks coordinate: mean S >= 0.005 with 90% lower bound > 0 in >= 75% of included markets.
  M4 stocks show more coordination than portfolios of the same region and dates: S_mean(stocks) >
     S_mean(portfolios) in >= 70% of the comparisons.
  M5 breadth: |S - S_pred| <= 0.015 in >= 75% of included markets.
  M6 the memory gap ranks the markets: Spearman(S_mean, S_pred) >= 0.7 over included markets.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments import e66_market_volatility as e66
from experiments import e74_nonlinear_gap as e74
from experiments.e76_leak_audit import series_shares

SEED = 79
SIM_SEEDS = tuple(range(910, 915))
SIM_CELLS = (("independent", 0.0), ("factor", 0.55), ("factor", 0.8), ("spill", 0.55))
MIN_MEMBERS = 15
COMPARATOR = {"US": "industries", "Japan": "Japan", "UK": "Europe", "Germany": "Europe", "France": "Europe",
              "Hong Kong": "Asia_Pacific_ex_Japan", "Australia": "Asia_Pacific_ex_Japan"}
VARS = ("mean", "breadth")


# ---------------------------------------------------------------- simulations

def garch(z, a=0.08, b=0.9, shocks=None):
    t, n = z.shape
    w = 1 - a - b
    h, r = np.ones(n), np.zeros((t, n))
    for k in range(t):
        r[k] = np.sqrt(h) * z[k]
        h = w + a * (r[k] ** 2 if shocks is None else shocks(r[k])) + b * h
    return r


def sim_market(seed, cond, load, t=6700, n=400):
    """Daily returns of n stocks: a GARCH common factor with the given loading plus own GARCH noise; 'spill' lets
    every stock's variance respond to the cross-sectional mean squared shock."""
    rng = np.random.default_rng([SEED, seed])
    f = garch(rng.standard_normal((t, 1)))
    spill = (lambda r: 0.5 * r ** 2 + 0.5 * (r ** 2).mean()) if cond == "spill" else None
    idio = garch(rng.standard_normal((t, n)), shocks=spill)
    return idio if cond == "independent" else load * f + np.sqrt(1 - load ** 2) * idio


def _group(r, dow, rng):
    l = np.log(np.abs(r) + e74.OFFSET)
    out = series_shares(l, dow, rng)
    out["virtual"] = {v: [] for v in VARS}
    for _ in range(e66.VIRTUAL):
        s = series_shares(e66.virtual(l, rng), dow, rng, full=False)
        for v in VARS:
            out["virtual"][v].append(s[v]["S"])
    return out


def _sim(job):
    seed, cond, load = job
    r = sim_market(seed, cond, load)
    rng = np.random.default_rng([SEED, seed, int(100 * load), len(cond)])
    dow = e66._dow(len(r))
    stocks = r[:, rng.choice(r.shape[1], 40, replace=False)]
    port = r[:, rng.permutation(r.shape[1])].reshape(len(r), 25, 16).mean(2)
    return {"seed": seed, "cond": cond, "load": load, "stocks": _group(stocks, dow, rng),
            "portfolios": _group(port, dow, rng)}


# ---------------------------------------------------------------- real markets

def _market(m):
    from experiments import stock_data as sd
    dates, r, names, info = sd.panel(m)
    out = {"market": m, "index": sd.PAGES[m][0], "names": names, **info,
           "first": int(dates[0]), "last": int(dates[-1])}
    if len(names) < MIN_MEMBERS:
        return out | {"included": False}
    rng = np.random.default_rng([SEED, len(m), sd.MARKETS.index(m)])
    return out | {"included": True, **_group(r, e66._dow_dates(dates), rng)}


def _portfolios(name):
    from experiments import finance_data as fd
    from experiments import stock_data as sd
    if name == "industries":
        dates, r, names = fd.industries(start=sd.START, end=sd.END)
    else:
        dates, r, names = fd.regional(name)
        keep = (dates >= sd.START) & (dates <= sd.END)
        dates, r = dates[keep], r[keep]
    rng = np.random.default_rng([SEED, len(name), 7])
    return {"set": name, "members": len(names), "days": len(dates), "first": int(dates[0]), "last": int(dates[-1]),
            **_group(r, e66._dow_dates(dates), rng)}


# ---------------------------------------------------------------- predictions

def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def predictions(res):
    sims, inc = res["sims"], [m for m in res["markets"] if m["included"]]
    port = {p["set"]: p for p in res["portfolios"]}
    out = []
    dep = [(s[g]["mean"]["S"], s[g]["mean"]["S_pred"]) for s in sims if s["cond"] != "independent"
           for g in ("stocks", "portfolios")]
    s1 = sum(abs(a - b) <= 0.005 for a, b in dep)
    out.append(("S1", "simulations: |S - S_pred| <= 0.005 (mean) in >= 90% of dependent cells", s1 >= 0.9 * len(dep),
                f"{s1}/{len(dep)}; worst {max(abs(a - b) for a, b in dep):.4f}"))
    ind = [s[g]["mean"]["S"] for s in sims if s["cond"] == "independent" for g in ("stocks", "portfolios")]
    virt = [v for s in sims for g in ("stocks", "portfolios") for v in s[g]["virtual"]["mean"]]
    out.append(("S2", "simulations: independent |S| <= 0.003 and every virtual |S| <= 0.003",
                max(map(abs, ind)) <= 0.003 and max(map(abs, virt)) <= 0.003,
                f"independent max {max(map(abs, ind)):.4f}; virtual max {max(map(abs, virt)):.4f}"))
    strong = [s for s in sims if s["cond"] == "factor" and s["load"] == 0.8]
    s3 = sum(s["stocks"]["mean"]["S"] > s["portfolios"]["mean"]["S"] for s in strong)
    out.append(("S3", "simulations: strong factor, stocks > portfolios in >= 4/5 seeds", s3 >= 4, f"{s3}/{len(strong)}; " +
                ", ".join(f"{s['stocks']['mean']['S']:+.4f}/{s['portfolios']['mean']['S']:+.4f}" for s in strong)))
    gaps = [m["mean"]["S"] - m["mean"]["S_pred"] for m in inc]
    out.append(("M1", "the memory gap holds for stocks: |S - S_pred| <= 0.01 (mean) in every included market",
                all(abs(g) <= 0.01 for g in gaps), ", ".join(f"{m['market']} {g:+.4f}" for m, g in zip(inc, gaps))))
    rv = [v for m in inc for v in m["virtual"]["mean"]]
    m2 = sum(abs(v) <= 0.01 for v in rv)
    out.append(("M2", "virtual stock groups clean: |S| <= 0.01 (mean) for >= 90%", m2 >= 0.9 * len(rv),
                f"{m2}/{len(rv)}; range {min(rv):+.4f} to {max(rv):+.4f}"))
    m3 = [m for m in inc if m["mean"]["S"] >= 0.005 and m["mean"]["lo"] > 0]
    out.append(("M3", "stocks coordinate: mean S >= 0.005, lower > 0 in >= 75% of markets", len(m3) >= 0.75 * len(inc),
                f"{len(m3)}/{len(inc)}; " + ", ".join(f"{m['market']} {m['mean']['S']:+.4f} [{m['mean']['lo']:+.4f}]"
                                                      for m in inc)))
    comp = [(m, port[COMPARATOR[m["market"]]]) for m in inc if m["market"] in COMPARATOR]
    m4 = sum(m["mean"]["S"] > p["mean"]["S"] for m, p in comp)
    out.append(("M4", "stocks > portfolios of the same region and dates in >= 70% of comparisons",
                m4 >= 0.7 * len(comp), f"{m4}/{len(comp)}; " + ", ".join(
                    f"{m['market']} {m['mean']['S']:+.4f} vs {p['mean']['S']:+.4f}" for m, p in comp)))
    bg = [m["breadth"]["S"] - m["breadth"]["S_pred"] for m in inc]
    m5 = sum(abs(g) <= 0.015 for g in bg)
    out.append(("M5", "breadth: |S - S_pred| <= 0.015 in >= 75% of markets", m5 >= 0.75 * len(inc),
                f"{m5}/{len(inc)}; " + ", ".join(f"{m['market']} {g:+.4f}" for m, g in zip(inc, bg))))
    rho = _spearman([m["mean"]["S"] for m in inc], [m["mean"]["S_pred"] for m in inc])
    out.append(("M6", "the memory gap ranks the markets: Spearman >= 0.7", rho >= 0.7, f"rho {rho:+.2f}"))
    return out


def analyze(res):
    lines = ["e79: individual stocks in eight markets", "", "Simulations (mean S / S_pred; stocks, portfolios):"]
    for s in res["sims"]:
        lines.append(f"  {s['seed']} {s['cond']:11s} {s['load']:.2f}  " + "  ".join(
            f"{g} {s[g]['mean']['S']:+.4f}/{s[g]['mean']['S_pred']:+.4f}" for g in ("stocks", "portfolios")))
    lines += ["", "Markets (members, days; mean S [90%], S_pred; breadth S, S_pred; virtual mean):"]
    for m in res["markets"]:
        head = f"  {m['market']:10s} {m['index']:11s} {m['members']:3d} of {m['listed']:3d}, {m['days']} days"
        if not m["included"]:
            lines.append(head + "  (excluded: too few members)")
            continue
        lines.append(head + f"  mean {m['mean']['S']:+.4f} [{m['mean']['lo']:+.4f}, {m['mean']['hi']:+.4f}] "
                     f"pred {m['mean']['S_pred']:+.4f}  breadth {m['breadth']['S']:+.4f} pred {m['breadth']['S_pred']:+.4f}"
                     f"  virtual {min(m['virtual']['mean']):+.4f}..{max(m['virtual']['mean']):+.4f}")
    lines += ["", "Portfolios, same dates:"]
    for p in res["portfolios"]:
        lines.append(f"  {p['set']:22s} {p['members']:3d}, {p['days']} days  mean {p['mean']['S']:+.4f} "
                     f"[{p['mean']['lo']:+.4f}, {p['mean']['hi']:+.4f}] pred {p['mean']['S_pred']:+.4f}  "
                     f"breadth {p['breadth']['S']:+.4f} pred {p['breadth']['S_pred']:+.4f}")
    lines += ["", "Predictions:"]
    preds = predictions(res)
    for k, lab, ok, d in preds:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {k} {lab}\n         {d}")
    lines.append(f"\n{sum(p[2] for p in preds)}/{len(preds)} predictions pass")
    return "\n".join(lines)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    from experiments import stock_data as sd
    sims = [(s, c, l) for s in SIM_SEEDS[:1 if quick else None] for c, l in SIM_CELLS]
    markets = sd.MARKETS[:2] if quick else sd.MARKETS
    sets = sorted(set(COMPARATOR.values()))
    with ProcessPoolExecutor(max_workers=7) as pool:
        res = {"sims": list(pool.map(_sim, sims)), "markets": list(pool.map(_market, markets)),
               "portfolios": list(pool.map(_portfolios, sets))}
    (out_dir / "e79_stocks_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e79_figures import make_figures
    make_figures(res, out_dir / "e79_figures", text)
    return text


if __name__ == "__main__":
    print(main())
