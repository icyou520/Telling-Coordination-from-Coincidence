"""Experiment 84: can memory find primes? Block histories as a guide to where to search.

A proposal: primes are not predictable one by one, but a search would gain if some places could be known in advance
to hold more of them. The memory-gap framework needs members, a group and a forecast, so:
    members   the number of primes ending in 1, 3, 7 and 9 in each block of B = 30030 consecutive integers
              (prime_data; blocks aligned so every block has identical divisibility by 2-13);
    group     the total per block;
    forecast  the next block's count from the counts of the blocks before it (the number line plays the role of time).
This is a forecasting comparison, not a claim that primes interact: the structure of primes is arithmetic.

The bar is an informed baseline, not blind search: trial division by every prime <= 101, and the prime number
theorem's density among the survivors (prime_data.expected). Any method must beat that, out of sample, and pay for
itself in primality tests saved.

Measures, per range of 5,000 blocks:
  calibration   total primes / baseline expectation;
  z             (count - expected) / sqrt(expected) per block (0 = exactly the baseline);
  memory gain   out-of-sample MSE gain of forecasting z from its past (HAR windows 1, 5, 22 blocks) over the baseline
                (forecast 0); coefficients fitted on the development range and frozen, and also refitted within each
                range by 5 contiguous folds (the most favourable case);
  search gain   primes per primality test in the 10% of blocks ranked highest by a score, relative to all blocks
                (testing every survivor of the 101-filter). Scores: the frozen memory forecast; an oracle that knows
                each block's true count (the ceiling for any block-ranking method); for scale, the gain from simply
                filtering deeper (trial division up to 1000 instead of 101). Noise: s.d. of the gain over 2,000
                random 10% selections;
  twinshare     the interaction share and checks of the four digit classes (members: counts; recorded causes: their
                baseline expectations), twinshare defaults.
Controls (simulated, each range's own survivors): "sieve-random" primes, each survivor prime independently at the
baseline rate (structure from divisibility only); "planted", the same with a slowly varying rate (AR(1), phi 0.9,
s.d. 3%), which memory should find.

Development: 10^9 (5,000 blocks from 1,000,029,030). Held out: 10^12 and 10^16, untouched before the predictions
below were hashed (a sieve check printed one calibration ratio over 300 blocks at 10^15, so 10^15 is not used).
The sieve agreed with a deterministic Miller-Rabin test on every integer of two blocks at 10^9, 10^12 and 10^15.
Development findings (10^9):
  * baseline calibrated (0.9999); 2.48 primality tests per prime after trial division to 101, 1.69 to 1000;
  * primes are more regular than divisibility alone makes them: block variance 0.43 of Poisson against 0.58 for
    sieve-random, and neighbouring blocks anti-correlated (lag 1: -0.064; sieve-random 0.000). This is the
    Hardy-Littlewood / Montgomery-Soundararajan picture (sub-Poisson counts in short intervals, negative
    correlation between neighbours, fading as x grows: roughly -ln 2 / (ln(x/B) - 3.4));
  * that memory is real but tiny: forecast gain 0.5% (refitted), search gain +0.24% (3 random s.d.), against
    +3.0% for the oracle and +46% for trial division to 1000;
  * twinshare: primes S +0.0003, "No coordination detected"; sieve-random -0.0009 (same); planted +0.068, detected;
  * the frozen coefficients do not transfer to the planted control (a different pattern), hence its refitted
    search gain is the power check.

Pre-registered predictions (both held-out ranges, 10^12 and 10^16):
  Controls
  C1  Sieve-random: twinshare "No coordination detected", and both memory search gains within 3 random s.d. of 0.
  C2  Planted: twinshare detects coordination, and the refitted memory search gain exceeds 3 random s.d.
  Primes
  P1  The baseline is calibrated: primes / expected within 0.002 of 1.
  P2  Primes are more regular than divisibility alone: block variance ratio below sieve-random's.
  P3  Faint memory: lag-1 autocorrelation of z negative in both ranges, and weaker at 10^16 than at 10^9 (-0.064).
  P4  No useful forecast: memory gain (frozen and refitted) below 0.01.
  P5  No useful search advantage: memory search gains (frozen and refitted) below +1%.
  P6  Divisibility dominates: trial division to 1000 gains more than the oracle ceiling of block ranking.
  P7  twinshare on the digit classes: "No coordination detected", |S| < 0.005.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments import prime_data as pd

SEED = 84
BLOCKS = 5000
DEV = 1e9
TESTS = (1e12, 1e16)
WINDOWS = (1, 5, 22)
LIM, DEEP = pd.FILTERS
TOP = 0.10
N_RANDOM = 2000
PLANT = (0.9, 0.03)
FOLDS = 5


# ---------------------------------------------------------------- forecasting from block histories

def z_score(count, expected):
    return (count - expected) / np.sqrt(expected)


def har(z, windows=WINDOWS):
    """Features for rows lead..T-1 (each from the blocks before it): means of the last w values of z."""
    lead = max(windows)
    c = np.concatenate([[0.0], np.cumsum(z)])
    t = np.arange(lead, len(z))
    x = np.column_stack([(c[t] - c[t - w]) / w for w in windows])
    return x, z[lead:]


def fit(x, y):
    a = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(a, y, rcond=None)[0]


def predict(x, coef):
    return coef[0] + x @ coef[1:]


def mse_gain(y, pred):
    return float(1 - np.mean((y - pred) ** 2) / np.mean(y ** 2))


def crossfit(x, y, folds=FOLDS, purge=max(WINDOWS)):
    pred = np.empty_like(y)
    edges = np.linspace(0, len(y), folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        train = np.ones(len(y), bool)
        train[max(a - purge, 0): b + purge] = False
        pred[a:b] = predict(x[a:b], fit(x[train], y[train]))
    return pred


# ---------------------------------------------------------------- the search

def search_gain(count, surv, score, top=TOP):
    k = int(round(top * len(score)))
    sel = np.argsort(-score, kind="stable")[:k]
    return float(count[sel].sum() / surv[sel].sum() / (count.sum() / surv.sum()) - 1)


def random_sd(count, surv, rng, top=TOP, n=N_RANDOM):
    return float(np.std([search_gain(count, surv, rng.random(len(count)), top) for _ in range(n)]))


# ---------------------------------------------------------------- one range

def twinshare_result(counts, expected, seed):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twinshare" / "src"))
    from twinshare import analysis, data

    r = analysis.analyze(data.from_arrays(counts.astype(float), expected, members=[f"digit {d}" for d in pd.DIGITS]),
                         seed=seed, detail=False)
    return {"S": r.S, "interval": list(r.interval), "memory_gap": r.memory_gap["S_pred"], "verdict": r.verdict,
            "null_bar": analysis.null_bar(r), "virtual": r.virtual,
            "checks": [{"name": c.name, "status": c.status, "value": c.value} for c in r.checks]}


def measure(count, surv, expected, deep_surv, coef, rng, seed, ts=True):
    """count, surv, expected: (blocks, 4) per digit class; deep_surv: survivors of the deeper filter."""
    tot, sv, ex = count.sum(1), surv.sum(1), expected.sum(1)
    z = z_score(tot, ex)
    x, y = har(z)
    lead = max(WINDOWS)
    c, s = tot[lead:], sv[lead:]
    refit = crossfit(x, y)
    out = {"calibration": float(tot.sum() / ex.sum()), "var_ratio": float(np.var(z)),
           "acf": [float(np.corrcoef(z[:-k], z[k:])[0, 1]) for k in range(1, 11)],
           "gain_frozen": mse_gain(y, predict(x, coef)), "gain_crossfit": mse_gain(y, refit),
           "search_memory": search_gain(c, s, predict(x, coef)), "search_crossfit": search_gain(c, s, refit),
           "search_oracle": search_gain(c, s, c / s),
           "search_random_sd": random_sd(c, s, rng),
           "deep_filter": float((tot.sum() / deep_surv.sum(1).sum()) / (tot.sum() / sv.sum()) - 1),
           "tests_per_prime": float(sv.sum() / tot.sum()),
           "tests_per_prime_deep": float(deep_surv.sum(1).sum() / tot.sum())}
    if ts:
        out["twinshare"] = twinshare_result(count, expected, seed)
    return out


def simulate(p, rng, plant=None):
    """Counts drawn from each range's survivors: each survivor prime with the baseline rate (times a slowly varying
    factor if planted)."""
    surv, ex = p[f"surv{LIM}"], p[f"expected{LIM}"]
    rate = ex / surv
    if plant:
        phi, sd = plant
        u = np.zeros(len(surv))
        e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), len(surv))
        u[0] = rng.normal(0, sd)
        for t in range(1, len(u)):
            u[t] = phi * u[t - 1] + e[t]
        rate = rate * (1 + u)[:, None]
    return rng.binomial(surv, np.clip(rate, 0, 1))


def run_range(x, coef, seed, blocks=BLOCKS, ts=True):
    rng = np.random.default_rng([SEED, seed])
    p = pd.panel(x, blocks)
    args = (p[f"surv{LIM}"], p[f"expected{LIM}"], p[f"surv{DEEP}"])
    res = {"x": x, "x0": p["x0"], "blocks": blocks, "primes": int(p["count"].sum()),
           "real": measure(p["count"], *args, coef, rng, seed, ts)}
    for name, plant in (("sieve_random", None), ("planted", PLANT)):
        res[name] = measure(simulate(p, rng, plant), *args, coef, rng, seed, ts)
    return res, p


def develop(quick=False):
    blocks = 1000 if quick else BLOCKS
    p = pd.panel(DEV, blocks)
    z = z_score(p["count"].sum(1), p[f"expected{LIM}"].sum(1))
    coef = fit(*har(z))
    dev, _ = run_range(DEV, coef, 0, blocks)
    dev["coef"] = coef.tolist()
    return dev, coef


def main(out_dir: Path | None = None, quick: bool = False, dev_only: bool = False) -> str:
    out_dir = out_dir or Path("results")
    dev, coef = develop(quick)
    res = {"dev": dev, "tests": []}
    if not dev_only:
        for i, x in enumerate(TESTS):
            res["tests"].append(run_range(x, coef, i + 1, 1000 if quick else BLOCKS)[0])
    if not dev_only:
        (out_dir / "e84_primes_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    if not dev_only:
        from experiments.e84_figures import make_figures
        make_figures(res, out_dir / "e84_figures", text)
    return text


# ---------------------------------------------------------------- report

def describe(name, m):
    t = m.get("twinshare")
    lines = [f"    {name:13s} calibration {m['calibration']:.4f}  var/Poisson {m['var_ratio']:.3f}  "
             f"acf1-3 {m['acf'][0]:+.3f} {m['acf'][1]:+.3f} {m['acf'][2]:+.3f}",
             f"    {'':13s} memory gain frozen {m['gain_frozen']:+.4f}, refitted {m['gain_crossfit']:+.4f}   "
             f"search gain: memory {m['search_memory']:+.4f} (refitted {m['search_crossfit']:+.4f}), "
             f"oracle {m['search_oracle']:+.4f}, "
             f"random s.d. {m['search_random_sd']:.4f}"]
    if t:
        lines.append(f"    {'':13s} twinshare S {t['S']:+.4f} [{t['interval'][0]:+.4f}, {t['interval'][1]:+.4f}], "
                     f"memory gap {t['memory_gap']:+.4f}, null bar {t['null_bar']:+.4f}: {t['verdict']}")
    return lines


def analyze(res) -> str:
    lines = ["Experiment 84: can memory find primes? Block histories against a divisibility-and-density baseline", ""]
    for label, r in [("development", res["dev"])] + [("held out", r) for r in res["tests"]]:
        real = r["real"]
        lines += [f"  {label}: x0 = {r['x0']:,} ({r['blocks']} blocks of {pd.B}, {r['primes']:,} primes); "
                  f"primality tests per prime found: {real['tests_per_prime']:.2f} after trial division to {LIM}, "
                  f"{real['tests_per_prime_deep']:.2f} to {DEEP} (gain {real['deep_filter']:+.3f})"]
        for name in ("real", "sieve_random", "planted"):
            lines += describe(name, r[name])
        lines.append("")
    if res["tests"]:
        lines.append("  Pre-registered predictions:")
        for key, label, ok, detail in predictions(res):
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


NONE = "No coordination detected"


def predictions(res):
    tests, dev = res["tests"], res["dev"]["real"]
    real = [t["real"] for t in tests]
    sr = [t["sieve_random"] for t in tests]
    pl = [t["planted"] for t in tests]
    fmt = lambda v, f="+.4f": ", ".join(format(x, f) for x in v)  # noqa: E731
    out = []
    ok = all(m["twinshare"]["verdict"] == NONE and abs(m["search_memory"]) < 3 * m["search_random_sd"]
             and abs(m["search_crossfit"]) < 3 * m["search_random_sd"] for m in sr)
    out.append(("C1", "sieve-random: no coordination, memory search gains within 3 s.d. of 0", ok,
                "; ".join(f"{m['twinshare']['verdict']}, {m['search_memory']:+.4f}/{m['search_crossfit']:+.4f} "
                          f"(s.d. {m['search_random_sd']:.4f})" for m in sr)))
    ok = all(m["twinshare"]["verdict"].startswith("Coordination detected")
             and m["search_crossfit"] > 3 * m["search_random_sd"] for m in pl)
    out.append(("C2", "planted: coordination detected, refitted search gain > 3 s.d.", ok,
                "; ".join(f"{m['twinshare']['verdict']}, {m['search_crossfit']:+.4f} (s.d. {m['search_random_sd']:.4f})"
                          for m in pl)))
    cal = [m["calibration"] for m in real]
    out.append(("P1", "baseline calibrated within 0.002", all(abs(c - 1) <= 0.002 for c in cal), fmt(cal)))
    out.append(("P2", "block variance below sieve-random's", all(m["var_ratio"] < s["var_ratio"] for m, s in zip(real, sr)),
                "; ".join(f"{m['var_ratio']:.3f} vs {s['var_ratio']:.3f}" for m, s in zip(real, sr))))
    a1 = [m["acf"][0] for m in real]
    out.append(("P3", "lag-1 autocorrelation negative, and weaker at 10^16 than 10^9",
                all(a < 0 for a in a1) and abs(a1[-1]) < abs(dev["acf"][0]),
                f"{fmt(a1)} (10^9 {dev['acf'][0]:+.4f})"))
    g = [v for m in real for v in (m["gain_frozen"], m["gain_crossfit"])]
    out.append(("P4", "memory forecast gains (frozen, refitted) < 0.01", max(g) < 0.01, fmt(g)))
    s = [v for m in real for v in (m["search_memory"], m["search_crossfit"])]
    out.append(("P5", "memory search gains (frozen, refitted) < +1%", max(s) < 0.01, fmt(s)))
    out.append(("P6", "trial division to 1000 beats the oracle ceiling",
                all(m["deep_filter"] > m["search_oracle"] for m in real),
                "; ".join(f"{m['deep_filter']:+.3f} vs {m['search_oracle']:+.3f}" for m in real)))
    S = [m["twinshare"]["S"] for m in real]
    out.append(("P7", "twinshare: no coordination, |S| < 0.005",
                all(m["twinshare"]["verdict"] == NONE for m in real) and max(map(abs, S)) < 0.005,
                f"S {fmt(S)}; " + "; ".join(m["twinshare"]["verdict"] for m in real)))
    return out


if __name__ == "__main__":
    main(dev_only="--dev" in sys.argv)
