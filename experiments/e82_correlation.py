"""Experiment 82: does the interaction share forecast next month's correlation between firms? A confirmation on fresh
markets.

e81 found, out of sample, that adding the share S to HAR + Diebold-Yilmaz improved forecasts of next month's mean
correlation between firms in all eight of e79's markets, with a negative sign (higher S, lower correlation later)
that was not predicted. e81 did not control for the current correlation, which is persistent.

Discovery (e81's eight markets, exploratory, before this file was hashed). Adding the current return correlation
(last 21 days, last 63 days, the 500-day window) to HAR + DY:
  S's out-of-sample gain fell from +0.3..+12.9% to: France +6.4, Germany +2.8, Japan +2.1, Australia +0.2, Hong Kong
  -0.4, Canada -0.6, UK -0.9, US -1.0 (%). The coefficient stayed negative in 7 of 8 (Australia positive); per s.d.
  of S, -0.008 to -0.038. The placebo S_v (time-shifted firms) under the same controls: coefficients split 4/4 in
  sign, -0.008 to +0.014, gains -1.2..+0.9%. So the e81 result was largely the missing control, with a residual
  negative effect concentrated in three markets.

Confirmation data: markets and firms never used before (experiments/stock_data_fresh.py): SMI, AEX, IBEX 35,
FTSE MIB, OMX Stockholm 30, OMX Helsinki 25, OBX, BEL 20, NIFTY 50, KOSPI 200, STI, S&P/NZX 50, B3, and S&P 500
members outside the S&P 100. e79's panel rule, unchanged; a market is used if it has >= 15 members (N markets).

Per market, e81's rolling windows (500 days, stepped 21; S, DY, S_v, HAR terms, target y_corr = mean pairwise
correlation of returns over the next 21 days), plus controls rc21, rc63, rcW = mean pairwise return correlation
over the last 21, 63 and 500 days. "Full" model: HAR + DY + rc21 + rc63 + rcW.

Pre-registered predictions (fresh markets only; discovery numbers are reported, not scored):
  F1  Pooled in-sample coefficient of S (standardized within market, market means removed) beyond the full model:
      90% moving-block bootstrap interval (blocks of 12 windows) entirely below 0.
  F2  The per-market coefficient of S beyond the full model is negative in >= 2/3 of the N markets.
  F3  Out of sample (expanding OLS from window 100, as e81): adding S to the full model lowers the MSE in at least
      half of the N markets and in the sum over markets.
  F4  Placebo: the pooled coefficient of S_v beyond the full model has a 90% interval that includes 0.
  F5  S beats its placebo: S's out-of-sample gain beyond the full model exceeds S_v's in more than half of the N
      markets.
  F6  e81's R6 on fresh data, without correlation controls: adding S to HAR + DY lowers the MSE in at least half
      of the N markets and in the sum.
Expectation from discovery: F1, F2, F4 and F6 likely; F3 and F5 uncertain (3-4 of 8 in discovery).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import e74_nonlinear_gap as e74
from experiments import e81_spillover as e81

SEED = 82
CTRL = ("rc21", "rc63", "rcW")
MIN_MEMBERS = 15
BLOCK = 12
N_BOOT = 400


def add_controls(rows, r):
    for w in rows:
        e = w["end"]
        w["rc21"] = e81.mean_corr(r[e - 21:e])
        w["rc63"] = e81.mean_corr(r[e - 63:e])
        w["rcW"] = e81.mean_corr(r[e - e81.W:e])
    return rows


def _x(rows, cols):
    return np.column_stack([np.ones(len(rows)), [w["har"] for w in rows]] + [[w[c] for w in rows] for c in cols])


def _oos_mse(rows, cols):
    x, y = _x(rows, cols), np.array([w["y_corr"] for w in rows])
    err = [y[t] - x[t] @ np.linalg.lstsq(x[:t - 1], y[:t - 1], rcond=None)[0] for t in range(e81.OOS_START, len(rows))]
    return float(np.mean(np.square(err)))


def _coef(rows, cols, key):
    """OLS coefficient of key beyond cols, per standard deviation of key."""
    x, y = _x(rows, cols + [key]), np.array([w["y_corr"] for w in rows])
    return float(np.linalg.lstsq(x, y, rcond=None)[0][-1] * np.std([w[key] for w in rows]))


def market(panel_fn, m, rng, quick=False):
    dates, r, names, info = panel_fn(m)
    if len(names) < MIN_MEMBERS:
        return {"market": m, "included": False, "members": len(names)}
    if quick:
        r = r[:3200]
    l = np.log(np.abs(r) + e74.OFFSET)
    rows = add_controls(e81.rolling(l, r, rng)["windows"], r)
    full = ["DY", *CTRL]
    out = {"market": m, "included": True, "members": len(names), "first": int(dates[0]), "windows": rows,
           "coef_S": _coef(rows, full, "S"), "coef_Sv": _coef(rows, full, "S_v"),
           "mse": {"full": _oos_mse(rows, full), "full+S": _oos_mse(rows, full + ["S"]),
                   "full+S_v": _oos_mse(rows, full + ["S_v"]), "e81": _oos_mse(rows, ["DY"]),
                   "e81+S": _oos_mse(rows, ["DY", "S"])}}
    return out


def pooled(markets, key, rng, n_boot=N_BOOT):
    """Pooled coefficient of key beyond the full model; predictors standardized within market, market means removed."""
    xs, ys, idx = [], [], []
    for mk in markets:
        rows = mk["windows"]
        x = _x(rows, ["DY", *CTRL, key])[:, 1:]
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        y = np.array([w["y_corr"] for w in rows])
        xs.append(x)
        ys.append(y - y.mean())
        idx.append(np.arange(len(rows)))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(idx)
    coef = float(np.linalg.lstsq(x, y, rcond=None)[0][-1])
    nb = int(w.max()) // BLOCK + 1
    boots = []
    for _ in range(n_boot):
        sel = np.concatenate([np.flatnonzero(w // BLOCK == b) for b in rng.integers(0, nb, nb)])
        boots.append(np.linalg.lstsq(x[sel], y[sel], rcond=None)[0][-1])
    return {"coef": coef, "ci": [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))]}


def _gain(m, a, b):
    return 100 * (1 - m["mse"][b] / m["mse"][a])


def predictions(res):
    mk = [m for m in res["fresh"] if m["included"]]
    n = len(mk)
    p, pv = res["pooled"]["fresh"]["S"], res["pooled"]["fresh"]["S_v"]
    names = lambda f: ", ".join(f"{m['market']} {f(m):+.1f}" for m in mk)  # noqa: E731
    out = [("F1", "pooled coefficient of S beyond HAR + DY + current correlation: 90% interval below 0",
            p["ci"][1] < 0, f"{p['coef']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]")]
    neg = sum(m["coef_S"] < 0 for m in mk)
    out.append(("F2", "per-market coefficient of S negative in >= 2/3 of markets", 3 * neg >= 2 * n,
                f"{neg}/{n}; " + ", ".join(f"{m['market']} {m['coef_S']:+.3f}" for m in mk)))
    wins = sum(m["mse"]["full+S"] < m["mse"]["full"] for m in mk)
    tot = sum(m["mse"]["full+S"] for m in mk) < sum(m["mse"]["full"] for m in mk)
    out.append(("F3", "out of sample: S beyond the full model helps in >= half of markets and in total",
                2 * wins >= n and tot, f"{wins}/{n}, total {'lower' if tot else 'not lower'}; %: "
                + names(lambda m: _gain(m, "full", "full+S"))))
    out.append(("F4", "placebo: pooled coefficient of S_v, 90% interval includes 0", pv["ci"][0] <= 0 <= pv["ci"][1],
                f"{pv['coef']:+.4f} [{pv['ci'][0]:+.4f}, {pv['ci'][1]:+.4f}]"))
    beat = sum(_gain(m, "full", "full+S") > _gain(m, "full", "full+S_v") for m in mk)
    out.append(("F5", "S's out-of-sample gain exceeds its placebo's in more than half of markets", 2 * beat > n,
                f"{beat}/{n}; placebo %: " + names(lambda m: _gain(m, "full", "full+S_v"))))
    wins = sum(m["mse"]["e81+S"] < m["mse"]["e81"] for m in mk)
    tot = sum(m["mse"]["e81+S"] for m in mk) < sum(m["mse"]["e81"] for m in mk)
    out.append(("F6", "e81's test on fresh data (no correlation controls): S helps in >= half and in total",
                2 * wins >= n and tot, f"{wins}/{n}, total {'lower' if tot else 'not lower'}; %: "
                + names(lambda m: _gain(m, "e81", "e81+S"))))
    return out


def analyze(res) -> str:
    lines = ["Experiment 82: does the share forecast next month's correlation between firms? Fresh markets", ""]
    for sample, label in (("discovery", "Discovery (e81's markets; not scored)"), ("fresh", "Fresh markets (scored)")):
        lines += [f"  {label}:", "  market        firms windows  coef S  coef S_v | OOS gain %: S beyond full  "
                  "placebo  | S beyond HAR+DY (e81)"]
        for m in res[sample]:
            if not m["included"]:
                lines.append(f"  {m['market']:12}  excluded ({m['members']} members)")
                continue
            lines.append(f"  {m['market']:12}  {m['members']:<5d} {len(m['windows']):<7d} {m['coef_S']:+.3f}   "
                         f"{m['coef_Sv']:+.3f}   |   {_gain(m, 'full', 'full+S'):+5.1f}             "
                         f"{_gain(m, 'full', 'full+S_v'):+5.1f}    |   {_gain(m, 'e81', 'e81+S'):+5.1f}")
        for k, v in res["pooled"][sample].items():
            lines.append(f"  pooled coefficient of {k} beyond HAR + DY + current correlation: {v['coef']:+.4f} "
                         f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
        lines.append("")
    lines.append("  Pre-registered predictions (fresh markets):")
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from experiments import stock_data as sd
    from experiments import stock_data_fresh as sdf
    out_dir = out_dir or Path("results")
    res = {"discovery": [market(sd.panel, m, np.random.default_rng([SEED, 0, i]), quick)
                         for i, m in enumerate(sd.MARKETS)],
           "fresh": [market(sdf.panel, m, np.random.default_rng([SEED, 1, i]), quick)
                     for i, m in enumerate(sdf.MARKETS)]}
    rng = np.random.default_rng([SEED, 2])
    res["pooled"] = {s: {k: pooled([m for m in res[s] if m["included"]], k, rng) for k in ("S", "S_v")}
                     for s in ("discovery", "fresh")}
    (out_dir / "e82_correlation_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e82_figures import make_figures
    make_figures(res, out_dir / "e82_figures", text)
    return text


if __name__ == "__main__":
    main()
