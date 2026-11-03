"""Experiment 81: the interaction share against the Diebold-Yilmaz spillover index, head to head.

An economist's first question about the share: does it tell me anything my spillover index does not? The standard
tool is Diebold & Yilmaz's (2012) total spillover index: fit a VAR to the members, and report the share of each
member's forecast-error variance (generalized decomposition, horizon 10 days) due to shocks in the others. The
share (here its memory gap, e67, which matched the measured share within 0.008 in these markets in e79) needs no
joint model.

Data: e79's eight stock panels (17-40 firms each, 2000-2026), members log(|r| + 0.05), the market variable their
mean. Rolling windows of 500 days, stepped by 21 days; in each window:
  S     memory gap of the mean (e74._pred, HAR windows);
  DY    total spillover index, VAR(1) on the members (OLS, demeaned), generalized FEVD at horizon 10, in %;
  corr  mean pairwise correlation of the members' log volatility (a co-movement baseline);
  S_v   S on a virtual market (each firm shifted by its own random offset over the whole sample, e66.virtual): as
        smooth as S, with no coordination; a placebo.
Targets, after the window: (1) the market's mean log volatility over the next 21 days; (2) the mean pairwise
correlation of returns over the next 21 days (whether diversification fails). Controls: HAR terms of the market
variable at the window's end (yesterday, last 5, last 22 days).

Tests:
  in-sample  pooled over markets (market means removed, predictors standardized within market): coefficient of S in
             target ~ HAR + DY + S, 90% interval by moving-block bootstrap over windows (blocks of 12, shared by all
             markets);
  out of     per market, expanding OLS from window 100 on, fitted only on windows whose target is already observed;
  sample     models HAR + DY (A), HAR + DY + S (B), HAR + S (C), HAR + DY + S_v (P). Gain = 1 - MSE(model)/MSE(A).
Simulations (known answers; e66's GARCH markets, 40 members, 6,700 days, full sample): correlated shocks (members
hit at the same moment, no dynamic interaction), spillover (each member's variance responds to the others' shocks),
a common factor, and independence.

Development (simulations only, seeds 30-31), full sample, 40 members:
  independent S 0.000, DY 0.7; correlated shocks S 0.002, DY 43-45; spillover S 0.022, DY 0.6; factor S 0.009-0.011,
  DY 71-76. DY registers simultaneous co-movement and not the dynamic spillover, at VAR orders 1, 2, 5 and 22 and on
  a 5-day smoothed proxy (1.7 against 1.8 independent). The draft prediction "DY sees spillover" was therefore
  replaced by D2 below.
  In 500-day windows DY is inflated by estimation: independent 15-18 at order 1, 44 at order 5, 95 with 400 members;
  spillover windows 14-16, indistinguishable. S is biased upward too (independent median 0.005, up to 0.026) but
  spillover windows sit higher (median 0.017-0.019).
  Time per window: S 0.6 ms against DY 1.2 ms at 40 members (the draft "S < 1/10 of DY at 40 firms" was dropped),
  3 ms against 72 ms at 400.

Pre-registered predictions (simulations on fresh seeds 32-33):
  D1  Correlation is not interaction: with correlated shocks DY exceeds the independent market's by >= 5 points,
      while |S| <= 0.005.
  D2  Spillover is seen by S, not by DY: with spillover S >= 0.01, and DY within 2 points of the independent
      market's.
  D3  In 500-day windows (stepped 250 days), spillover's median S is above the independent market's, while its
      median DY is not more than 2 points above.
  D4  Cost at scale: with 400 members in a 500-day window, S takes < 1/10 of DY's time.
  R1  Not redundant: |Spearman(S, DY)| over windows < 0.9 in >= 6 of 8 markets.
  R2  In-sample, volatility: pooled coefficient of S beyond HAR + DY, 90% interval above 0.
  R3  Out of sample, volatility: adding S to HAR + DY lowers the MSE (B better than A) in >= 5 of 8 markets and in
      the sum over markets.
  R4  Out of sample, volatility, the reverse: adding DY to HAR + S lowers the MSE (B better than C) in >= 5 of 8
      markets. (Scored as stated; a FAIL means DY adds nothing beyond S.)
  R5  Placebo: the virtual S does not help, P not better than A in >= 5 of 8 markets.
  R6  Out of sample, correlation target: B better than A in >= 5 of 8 markets and in the sum.
The volatility proxy is daily log(|r| + 0.05); Diebold and Yilmaz used range-based volatility, which is less noisy
(Yahoo's adjusted closes give no range). Their index is computed as they define it otherwise.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments import e66_market_volatility as e66
from experiments import e74_nonlinear_gap as e74

SEED = 81
W = 500
STEP = 21
AHEAD = 21
H = 10
OOS_START = 100
BLOCK = 12
SIM_SEEDS = (32, 33)


# ---------------------------------------------------------------- the two measures

def dy_index(x, h=H):
    """Diebold-Yilmaz (2012) total spillover index (%) of x (days, members): VAR(1) by OLS on demeaned data,
    generalized forecast-error variance decomposition at horizon h, rows normalized."""
    x = x - x.mean(0)
    n = x.shape[1]
    a = np.linalg.lstsq(x[:-1], x[1:], rcond=None)[0].T
    e = x[1:] - x[:-1] @ a.T
    sig = e.T @ e / (len(e) - n - 1)
    num, den = np.zeros((n, n)), np.zeros(n)
    ah = np.eye(n)
    for _ in range(h):
        m = ah @ sig
        num += m ** 2
        den += np.einsum("ij,ij->i", m, ah)
        ah = a @ ah
    theta = num / np.diag(sig)[None, :] / den[:, None]
    theta /= theta.sum(1, keepdims=True)
    return float(100 * (theta.sum() - np.trace(theta)) / n)


def mean_corr(x):
    """Mean pairwise correlation over the pairs where it is defined (a firm with flat prices for a month has none)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(x.T)
    return float(np.nanmean(c[~np.eye(len(c), dtype=bool)]))


def rolling(l, r, rng, timing=False):
    """Per window ending at day e: S, DY, corr, S_v, HAR controls and the two targets."""
    lv = e66.virtual(l, rng)
    m = l.mean(1)
    rows, t_s, t_dy = [], 0.0, 0.0
    for end in range(W, len(l) - AHEAD + 1, STEP):
        x = l[end - W:end]
        t0 = time.perf_counter()
        s = e74._pred(x)
        t1 = time.perf_counter()
        dy = dy_index(x)
        t2 = time.perf_counter()
        t_s, t_dy = t_s + t1 - t0, t_dy + t2 - t1
        rows.append({"end": end, "S": s, "DY": dy, "corr": mean_corr(x), "S_v": e74._pred(lv[end - W:end]),
                     "har": [float(m[end - 1]), float(m[end - 5:end].mean()), float(m[end - 22:end].mean())],
                     "y_vol": float(m[end:end + AHEAD].mean()), "y_corr": mean_corr(r[end:end + AHEAD])})
    out = {"windows": rows}
    if timing:
        out["seconds"] = {"S": t_s, "DY": t_dy}
    return out


# ---------------------------------------------------------------- tests

def _design(rows, cols, z=None):
    har = np.array([r["har"] for r in rows])
    extra = np.column_stack([[r[c] for r in rows] for c in cols]) if cols else np.zeros((len(rows), 0))
    if z is not None:
        extra = (extra - z[0]) / z[1]
    return np.column_stack([np.ones(len(rows)), har, extra])


def oos(rows, target):
    """Out-of-sample MSE of each model with expanding OLS; window t's target is observed once window t+1 ends."""
    models = {"A": ["DY"], "B": ["DY", "S"], "C": ["S"], "P": ["DY", "S_v"], "base": []}
    y = np.array([r[target] for r in rows])
    out = {}
    for name, cols in models.items():
        x = _design(rows, cols)
        err = []
        for t in range(OOS_START, len(rows)):
            beta = np.linalg.lstsq(x[:t - 1], y[:t - 1], rcond=None)[0]
            err.append(y[t] - x[t] @ beta)
        out[name] = float(np.mean(np.square(err)))
    return out


def _std(rows, cols):
    v = np.column_stack([[r[c] for r in rows] for c in cols])
    return v.mean(0), v.std(0) + 1e-12


def pooled(markets, target, rng, n_boot=400):
    """Coefficient of S (standardized within market) in target ~ HAR + DY + S, market means removed."""
    xs, ys, idx = [], [], []
    for rows in markets:
        x = _design(rows, ["DY", "S"], _std(rows, ["DY", "S"]))[:, 1:]
        y = np.array([r[target] for r in rows])
        xs.append(x - x.mean(0))
        ys.append(y - y.mean())
        idx.append(np.arange(len(rows)))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(idx)
    coef = float(np.linalg.lstsq(x, y, rcond=None)[0][-1])
    nb = int(w.max()) // BLOCK + 1
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, nb, nb)
        sel = np.concatenate([np.flatnonzero(w // BLOCK == b) for b in pick])
        boots.append(np.linalg.lstsq(x[sel], y[sel], rcond=None)[0][-1])
    return {"coef": coef, "ci": [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))]}


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- runs

def sim_cell(cond, seed, n=40):
    l = np.log(e66.simulate(cond, seed, n=n) + e74.OFFSET)
    starts = range(0, len(l) - W, 250)
    return {"cond": cond, "seed": seed, "S": e74._pred(l), "DY": dy_index(l), "corr": mean_corr(l),
            "S_win": float(np.median([e74._pred(l[a:a + W]) for a in starts])),
            "DY_win": float(np.median([dy_index(l[a:a + W]) for a in starts]))}


def sim_timing(seed, n=400, reps=5):
    l = np.log(e66.simulate("spillover", seed, n=n, t=W + 100) + e74.OFFSET)[:W]
    t0 = time.perf_counter()
    for _ in range(reps):
        e74._pred(l)
    t1 = time.perf_counter()
    for _ in range(reps):
        dy_index(l)
    t2 = time.perf_counter()
    return {"members": n, "S": (t1 - t0) / reps, "DY": (t2 - t1) / reps}


def real_market(m, quick=False):
    from experiments import stock_data as sd
    dates, r, names, _ = sd.panel(m)
    if len(names) < 15:
        return {"market": m, "included": False}
    if quick:
        r = r[:3200]
    l = np.log(np.abs(r) + e74.OFFSET)
    rng = np.random.default_rng([SEED, sd.MARKETS.index(m)])
    out = rolling(l, r, rng, timing=True)
    rows = out["windows"]
    return {"market": m, "included": True, "members": len(names), **out,
            "spearman_S_DY": _spearman([w["S"] for w in rows], [w["DY"] for w in rows]),
            "spearman_S_corr": _spearman([w["S"] for w in rows], [w["corr"] for w in rows]),
            "oos": {t: oos(rows, t) for t in ("y_vol", "y_corr")} if len(rows) > OOS_START + 10 else None}


def predictions(res):
    sims = res["sim"]
    med = lambda cond, k: float(np.median([s[k] for s in sims if s["cond"] == cond]))  # noqa: E731
    out = []
    d_dy = med("correlated", "DY") - med("independent", "DY")
    out.append(("D1", "simulated correlated shocks: DY >= independent + 5 points, |S| <= 0.005",
                d_dy >= 5 and abs(med("correlated", "S")) <= 0.005,
                f"DY {med('correlated', 'DY'):.1f} vs {med('independent', 'DY'):.1f}; S {med('correlated', 'S'):+.4f}"))
    out.append(("D2", "simulated spillover: S >= 0.01, DY within 2 points of independent",
                med("spillover", "S") >= 0.01 and abs(med("spillover", "DY") - med("independent", "DY")) <= 2,
                f"S {med('spillover', 'S'):+.4f}; DY {med('spillover', 'DY'):.1f} vs {med('independent', 'DY'):.1f}"))
    out.append(("D3", "500-day windows: spillover's median S above independent's, median DY not > 2 points above",
                med("spillover", "S_win") > med("independent", "S_win")
                and med("spillover", "DY_win") - med("independent", "DY_win") <= 2,
                f"S {med('spillover', 'S_win'):+.4f} vs {med('independent', 'S_win'):+.4f}; DY "
                f"{med('spillover', 'DY_win'):.1f} vs {med('independent', 'DY_win'):.1f}"))
    tm = res["timing"]
    out.append(("D4", "cost at scale: 400 members, S < 1/10 of DY's time per window", tm["S"] < 0.1 * tm["DY"],
                f"S {1000 * tm['S']:.1f} ms, DY {1000 * tm['DY']:.1f} ms"))
    mk = [m for m in res["markets"] if m["included"]]
    rho = [m["spearman_S_DY"] for m in mk]
    out.append(("R1", "not redundant: |Spearman(S, DY)| < 0.9 in >= 6 of 8 markets", sum(abs(x) < 0.9 for x in rho) >= 6,
                ", ".join(f"{m['market']} {x:+.2f}" for m, x in zip(mk, rho))))
    p = res["pooled"]["y_vol"]
    out.append(("R2", "in-sample, volatility: pooled coefficient of S beyond HAR + DY, 90% interval above 0",
                p["ci"][0] > 0, f"{p['coef']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]"))
    for key, target, a, b, label in (("R3", "y_vol", "A", "B", "adding S to HAR + DY helps (volatility)"),
                                     ("R4", "y_vol", "C", "B", "adding DY to HAR + S helps (volatility)"),
                                     ("R6", "y_corr", "A", "B", "adding S to HAR + DY helps (future correlation)")):
        wins = [m["oos"][target][b] < m["oos"][target][a] for m in mk]
        tot_a, tot_b = sum(m["oos"][target][a] for m in mk), sum(m["oos"][target][b] for m in mk)
        ok = sum(wins) >= 5 and (tot_b < tot_a if key != "R4" else True)
        gains = ", ".join(f"{m['market']} {100 * (1 - m['oos'][target][b] / m['oos'][target][a]):+.1f}%" for m in mk)
        out.append((key, f"out of sample: {label}, in >= 5 of 8 markets" + ("" if key == "R4" else " and in total"),
                    ok, f"{sum(wins)}/8; {gains}"))
        if key == "R4":
            wins = [m["oos"]["y_vol"]["P"] >= m["oos"]["y_vol"]["A"] for m in mk]
            out.append(("R5", "placebo: the virtual S does not help (P not better than A) in >= 5 of 8 markets",
                        sum(wins) >= 5, f"{sum(wins)}/8; " + ", ".join(
                            f"{m['market']} {100 * (1 - m['oos']['y_vol']['P'] / m['oos']['y_vol']['A']):+.1f}%" for m in mk)))
    return out


def analyze(res) -> str:
    lines = ["Experiment 81: the interaction share against the Diebold-Yilmaz spillover index", "",
             "  Simulated markets (40 members): full sample, and medians over 500-day windows:",
             "  cond         seed  S        DY     corr    | S window  DY window"]
    for s in res["sim"]:
        lines.append(f"  {s['cond']:12} {s['seed']}  {s['S']:+.4f}  {s['DY']:5.1f}  {s['corr']:+.3f}  |  "
                     f"{s['S_win']:+.4f}   {s['DY_win']:5.1f}")
    t = res["timing"]
    lines.append(f"  time per 500-day window, {t['members']} members: S {1000 * t['S']:.1f} ms, DY {1000 * t['DY']:.1f} ms")
    lines += ["", "  Real markets (rolling 500-day windows):",
              "  market      firms windows  S med   DY med  rho(S,DY) rho(S,corr)  OOS gain vol: +S  +DY  placebo | corr: +S"]
    for m in res["markets"]:
        if not m["included"]:
            lines.append(f"  {m['market']:10}  excluded")
            continue
        w = m["windows"]
        o, c = m["oos"]["y_vol"], m["oos"]["y_corr"]
        g = lambda d, b, a: 100 * (1 - d[b] / d[a])  # noqa: E731
        lines.append(f"  {m['market']:10}  {m['members']:<5d} {len(w):<7d}  {np.median([x['S'] for x in w]):+.3f}  "
                     f"{np.median([x['DY'] for x in w]):5.1f}   {m['spearman_S_DY']:+.2f}      {m['spearman_S_corr']:+.2f}"
                     f"        {g(o, 'B', 'A'):+5.1f}% {g(o, 'B', 'C'):+5.1f}% {g(o, 'P', 'A'):+5.1f}%  | "
                     f"{g(c, 'B', 'A'):+5.1f}%")
    for t, v in res["pooled"].items():
        lines.append(f"  pooled in-sample coefficient of S beyond HAR + DY, {t}: {v['coef']:+.4f} "
                     f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from experiments import stock_data as sd
    out_dir = out_dir or Path("results")
    conds = ("independent", "correlated", "spillover", "factor")
    res = {"sim": [sim_cell(c, s) for c in conds for s in SIM_SEEDS[: 1 if quick else None]],
           "timing": sim_timing(SIM_SEEDS[0]),
           "markets": [real_market(m, quick) for m in sd.MARKETS]}
    mk = [m["windows"] for m in res["markets"] if m["included"]]
    rng = np.random.default_rng([SEED, 1])
    res["pooled"] = {t: pooled(mk, t, rng) for t in ("y_vol", "y_corr")}
    (out_dir / "e81_spillover_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e81_figures import make_figures
    make_figures(res, out_dir / "e81_figures", text)
    return text


if __name__ == "__main__":
    main()
