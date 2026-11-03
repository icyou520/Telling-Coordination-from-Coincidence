"""Experiment 89: time reversal. Does the share know which way time runs?

Run the same records backwards. Three measures:
    memory gap   s_pred from autocovariances, which are the same read forwards or backwards: the memory gap is
                 exactly invariant (a theorem for the biased estimator, checked here to rounding error);
    share        twinshare's cross-fitted share (members' twins, the group law): linear forecasts of a stationary
                 process are nearly reversible, but heterogeneous members can break the symmetry;
    influence    twinshare's per-member closure gain (how much one member's history improves the group's forecast):
                 the measure that should point at a leader forwards, and away from it backwards.
Data:
    common     10 members, AR(0.5) each plus a hidden shared AR(0.9) driver (no interaction; T 5,000);
    leader     member 0 is AR(0.9); the others x_i(t) = 0.5 x_i(t-1) + 0.4 x_0(t-1) + noise (T 5,000);
    markets    daily log volatility, log(|r| + 0.05), of the 30 Fama-French industries with day-of-week dummies as
               recorded causes (twinshare defaults). Development: value-weighted 1936-1971. Held out:
               equal-weighted 1972-2026 (analysed forwards in e76 and twinshare, never backwards).

Development (simulations seed 0; markets 1936-1971):
  * memory gap: forward and reversed equal to 3e-15 everywhere;
  * share: reversed minus forward +0.0025 (common driver, S 0.159), +0.0034 (leader, 0.234), +0.0006 (markets, 0.035);
  * influence: leader member 0 +0.625 forwards, +0.061 reversed (10x lower); common driver and markets: every
    member <= 0.0081 both ways.
  So the share measures how much coordination there is, not its direction; the direction is in the influence.

Pre-registered predictions (simulations seeds 1-3; markets 1972-2026):
  T1  Memory gap identical forwards and backwards (|difference| < 1e-10), every dataset.
  T2  Share nearly reversible: |S reversed - S forward| <= 0.01, every dataset.
  T3  Leader: forwards member 0 has the largest influence and it exceeds 0.3; reversed, it falls more than 5-fold.
  T4  Common driver: every member's influence < 0.01, both ways (no leader to find).
  T5  Markets 1972-2026: every industry's influence < 0.02, both ways.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twinshare" / "src"))
from twinshare import engine as E  # noqa: E402

SEED = 89
N, T = 10, 5000
DESIGN = E.Design(windows=(1, 5, 22))


def common(rng, n=N, t=T):
    d = np.zeros(t)
    x = np.zeros((t, n))
    for s in range(1, t):
        d[s] = 0.9 * d[s - 1] + rng.normal()
        x[s] = 0.5 * x[s - 1] + 0.5 * d[s] + rng.normal(size=n)
    return x, np.zeros((t, 0))


def leader(rng, n=N, t=T):
    x = np.zeros((t, n))
    for s in range(1, t):
        x[s, 0] = 0.9 * x[s - 1, 0] + rng.normal()
        x[s, 1:] = 0.5 * x[s - 1, 1:] + 0.4 * x[s - 1, 0] + rng.normal(size=n - 1)
    return x, np.zeros((t, 0))


def markets(which):
    from experiments import e66_market_volatility as e66
    from experiments import finance_data as fd
    title, start, end = {"dev": ("Average Value Weighted Returns", 19360101, 19711231),
                         "test": ("Average Equal Weighted Returns", 19720101, 99999999)}[which]
    dates, r, _ = fd.industries(title, start, end)
    return np.log(np.abs(r) + 0.05), np.eye(5)[e66._dow_dates(dates)][:, 1:]


def both_ways(x, z):
    out = {}
    for way, (xx, zz) in (("forward", (x, z)), ("reversed", (x[::-1].copy(), z[::-1].copy()))):
        err = E.share(xx, zz, DESIGN)
        out[way] = {"S": err.S, "interval": list(err.interval(np.random.default_rng(SEED))),
                    "memory_gap": E.memory_gap(xx, zz, DESIGN)["S_pred"],
                    "influence": E.influence(xx, zz, DESIGN).tolist()}
    return out


def describe(name, r):
    f, b = r["forward"], r["reversed"]
    lines = [f"  {name}"]
    for way, m in (("forward", f), ("reversed", b)):
        inf = np.array(m["influence"])
        lines.append(f"    {way:9s} S {m['S']:+.4f} [{m['interval'][0]:+.4f}, {m['interval'][1]:+.4f}]  memory gap "
                     f"{m['memory_gap']:+.6f}  influence: top member {int(inf.argmax())} ({inf.max():+.4f}), "
                     f"member 0 {inf[0]:+.4f}, median {np.median(inf):+.4f}")
    lines.append(f"    difference S {b['S'] - f['S']:+.4f}, memory gap {b['memory_gap'] - f['memory_gap']:+.1e}")
    return lines


SEEDS = (1, 2, 3)


def predictions(res):
    allr = list(res.values())
    out = []
    d = max(abs(r["reversed"]["memory_gap"] - r["forward"]["memory_gap"]) for r in allr)
    out.append(("T1", "memory gap identical both ways (< 1e-10)", d < 1e-10, f"max {d:.1e}"))
    ds = {k: r["reversed"]["S"] - r["forward"]["S"] for k, r in res.items()}
    out.append(("T2", "share nearly reversible (|difference| <= 0.01)", max(map(abs, ds.values())) <= 0.01,
                ", ".join(f"{k} {v:+.4f}" for k, v in ds.items())))
    lead = [r for k, r in res.items() if k.startswith("leader")]
    ok = all(int(np.argmax(r["forward"]["influence"])) == 0 and r["forward"]["influence"][0] > 0.3
             and r["reversed"]["influence"][0] < r["forward"]["influence"][0] / 5 for r in lead)
    out.append(("T3", "leader: top influence > 0.3 forwards, falls > 5-fold reversed", ok,
                "; ".join(f"{r['forward']['influence'][0]:+.3f} -> {r['reversed']['influence'][0]:+.3f}" for r in lead)))
    com = [r for k, r in res.items() if k.startswith("common")]
    m = max(max(r[w]["influence"]) for r in com for w in ("forward", "reversed"))
    out.append(("T4", "common driver: every influence < 0.01 both ways", m < 0.01, f"max {m:+.4f}"))
    mk = res["markets 1972-2026"]
    m = max(max(mk[w]["influence"]) for w in ("forward", "reversed"))
    out.append(("T5", "markets 1972-2026: every influence < 0.02 both ways", m < 0.02, f"max {m:+.4f}"))
    return out


def analyze(res):
    lines = ["Experiment 89: time reversal. Does the share know which way time runs?", ""]
    for k, r in res.items():
        lines += describe(k, r)
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False, data=("markets 1972-2026", "test")) -> str:
    out_dir = out_dir or Path("results")
    res = {}
    for s in ((101,) if quick else SEEDS):
        rng = np.random.default_rng([SEED, s])
        res[f"common driver, seed {s}"] = both_ways(*common(rng))
        res[f"leader, seed {s}"] = both_ways(*leader(rng))
    res[data[0]] = both_ways(*markets(data[1]))
    (out_dir / "e89_time_reversal_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e89_figures import make_figures
    make_figures(res, out_dir / "e89_figures", text)
    return text


if __name__ == "__main__":
    main()
