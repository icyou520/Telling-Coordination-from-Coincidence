"""Experiment 69: is there a small interaction share in market volatility after all? (e67's lead, confirmed on new data)

e66 found no coordination in US market volatility measured as daily |return| (S = -0.005). In e67 the memory-gap
equation predicted +0.016 there. An exploratory check after e67 found the cause: a few crash days dominate |return|.
On log volatility, log(|return| + 0.05), measured S = 0.012 against a predicted 0.015 (1972-2026, value-weighted).
Because that was found after looking, it is tested here on data not used before:
  holdout  1936-1971, value-weighted, the 40 industries with no missing day (never analysed before);
  equal    1972-2026, equal-weighted returns of the 49 industries (a different weighting of the same years).
The 1972-2026 value-weighted series is reported for reference only (already seen).

Pipeline: e66's cross-fitted HAR twin and law (day-of-week dummies; Saturdays before 1952 are merged with
Fridays), 250-day blocks in 3 folds, 400 block-bootstrap resamples, 5 virtual groups (circular shifts >= 500
days). S_pred: e67's memory-gap formula with HAR windows on the same series. finance_data was extended to read other
spans and sections; its default output (e66's data) is unchanged.

Predictions, fixed before running:
  M1  Holdout, log volatility: S >= 0.005 and 90% lower bound > 0.
  M2  Holdout, log volatility: |S - S_pred| <= 0.01.
  M3  Holdout, log volatility: all 5 virtual |S| <= 0.01.
  M4  Equal-weighted 1972-2026, log volatility: S >= 0.005 and 90% lower bound > 0.
  M5  The heavy-tail explanation: on the holdout, raw |return| gives S at least 0.005 below log volatility's S.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import e66_market_volatility as e66
from experiments.e67_memory_gap import HAR_WINDOWS, autocov, s_pred, series_design

SEED = 69
OFFSET = 0.05
SETS = {"holdout": ("Average Value Weighted Returns", 19360101, 19711231),
        "equal": ("Average Equal Weighted Returns", 19720101, 99999999),
        "reference": ("Average Value Weighted Returns", 19720101, 99999999)}


def _pred(x):
    offs, f, y = series_design(HAR_WINDOWS)
    xc = (x - x.mean(0)).T[:, None, :]
    o, g = autocov(xc, int(offs.max() - offs.min()))
    return s_pred(o, g, offs, f, y)


def run(name):
    from experiments import finance_data as fd
    title, start, end = SETS[name]
    dates, r, names = fd.industries(title, start, end)
    dow = e66._dow_dates(dates)
    rng = np.random.default_rng([SEED, len(name)])
    out = {"name": name, "members": len(names), "days": len(dates)}
    for kind, x in (("log", np.log(np.abs(r) + OFFSET)), ("raw", np.abs(r))):
        res = e66.summarize(*e66.share(x, dow), rng)
        res["virtual"] = [e66.summarize(*e66.share(e66.virtual(x, rng), dow), rng)["S"] for _ in range(e66.VIRTUAL)]
        res["S_pred"] = _pred(x)
        out[kind] = res
    return out


def predictions(res):
    h, e = res.get("holdout"), res.get("equal")
    hl = h["log"] if h else None
    return [
        ("M1", "holdout 1936-1971, log volatility: S >= 0.005, 90% lower bound > 0",
         bool(hl) and hl["S"] >= 0.005 and hl["lo"] > 0, f"S {hl['S']:+.4f} [{hl['lo']:+.4f}, {hl['hi']:+.4f}]" if hl else "n/a"),
        ("M2", "holdout: |S - S_pred| <= 0.01", bool(hl) and abs(hl["S"] - hl["S_pred"]) <= 0.01,
         f"S_pred {hl['S_pred']:+.4f}" if hl else "n/a"),
        ("M3", "holdout: all 5 virtual |S| <= 0.01", bool(hl) and all(abs(v) <= 0.01 for v in hl["virtual"]),
         " ".join(f"{v:+.4f}" for v in hl["virtual"]) if hl else "n/a"),
        ("M4", "equal-weighted 1972-2026, log volatility: S >= 0.005, lower bound > 0",
         bool(e) and e["log"]["S"] >= 0.005 and e["log"]["lo"] > 0,
         f"S {e['log']['S']:+.4f} [{e['log']['lo']:+.4f}, {e['log']['hi']:+.4f}]" if e else "n/a"),
        ("M5", "heavy tails: holdout raw |return| S at least 0.005 below log volatility S",
         bool(h) and h["raw"]["S"] <= hl["S"] - 0.005, f"raw {h['raw']['S']:+.4f} vs log {hl['S']:+.4f}" if h else "n/a"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 69: a small interaction share in market volatility? (log volatility, new data)", ""]
    for name in SETS:
        r = res[name]
        lines.append(f"  {name:9} {r['members']} industries, {r['days']} days")
        for kind in ("log", "raw"):
            k = r[kind]
            lines.append(f"    {kind:3}  S {k['S']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}]  S_pred {k['S_pred']:+.4f}  "
                         f"virtual {' '.join(f'{v:+.4f}' for v in k['virtual'])}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    res = {name: run(name) for name in SETS}
    (out_dir / "e69_market_log_volatility_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e69_figures import make_figures
    make_figures(res, out_dir / "e69_figures", text)
    return text


if __name__ == "__main__":
    main()
