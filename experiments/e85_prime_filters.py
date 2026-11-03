"""Experiment 85: memory on top of every filter. Does the faint memory of e84 survive deeper divisibility filters, and
does it pay once a search is charged for the information it needs?

e84 tested memory on top of trial division by primes <= 101: a real but tiny gain (at most +0.3% primes per
primality test), against +47% for filtering to 1000. Here the known filters are stacked first and memory is asked
for what is left:
    depth D       trial division by every prime <= D, for D = 101, 10^3, 10^4, 10^5, 10^6 (a search would stop
                  where one more filter costs more than the tests it saves; every depth here is below sqrt(x));
    baseline      the prime number theorem's count among the depth-D survivors (prime_data.panel);
    z_D           (primes - expected) / sqrt(expected) per block of 30030;
    memory        e84's forecast of z_D from the blocks before it (window means over 1, 5, 22 blocks), coefficients
                  fitted per depth on the development range and frozen (and refitted within each range);
    search gain   primes per primality test in the blocks a score selects (10%), relative to testing every
                  depth-D survivor:
                    free     the score may use every earlier block's true count (e84's best case);
                    charged  a real search knows a block's count only if it tested the block: blocks are visited in
                             order, the first 100 tested in full (warm-up, not scored), and block k is tested if its
                             forecast, from the known counts of earlier tested blocks (untested blocks count as the
                             baseline), is at or above the 90th percentile of all forecasts so far. Its noise is
                             the s.d. of random selections of the same number of blocks;
                    oracle   the block's true count (the ceiling for any way of choosing blocks).
Controls at depth 10^4: sieve-random primes (each survivor prime at the baseline rate) and the same with e84's
planted slow rate (AR(1), phi 0.9, s.d. 3%).

Development: 10^14 (5,000 blocks). Held out: 10^17 and 10^18 (5,000 blocks each), never analysed before the
predictions below were hashed (a timing check sieved 50 blocks at 10^18 and printed only the time).
Development findings (10^14):
  * tests per prime 3.84, 2.61, 1.96, 1.59, 1.29 from depth 101 to 10^6; the oracle ceiling shrinks 4.6% -> 2.7%;
  * memory gains at every depth within about 1 random s.d. of 0, free (-0.0009 to +0.0003) and charged (-0.0007 to
    +0.0011); lag-1 correlation -0.030 at depth 101, fading to 0 by 10^4;
  * the baseline drifts at deep filters (0.990 at 10^5, 1.017 at 10^6): near sqrt(x) the survivors' density follows
    Buchstab's function, omega(u) / ln D with u = ln x / ln D, not the Mertens product e^-gamma / ln D. For
    2 <= u <= 3, omega(u) = (1 + ln(u - 1)) / u, so the calibration should be e^-gamma / omega(u): 1.0174 at 10^6
    (measured 1.0171) and 0.9903 at 10^5 (0.9899);
  * a first version of the controls used the coefficients learned on real primes, which cannot find a planted
    pattern; the controls now learn theirs on one simulated development series (planted: charged gain +2.0%,
    13 s.d.; sieve-random: within 1.4 s.d.);
  * a structure check on stand-in ranges (3x10^14, 5x10^15, 1,000 blocks) showed the charged search tests 6-10% of
    blocks, not exactly 10%; its noise is now measured with random selections of the same size.

Pre-registered predictions (both held-out ranges, 10^17 and 10^18):
  C1  Sieve-random (depth 10^4): free, refitted and charged search gains within 3 random s.d. of 0.
  C2  Planted (depth 10^4): refitted and charged search gains above 3 random s.d. (the charged search has power).
  H1  Every tenfold deeper filter lowers tests per prime by more than the oracle ceiling at the shallower depth.
  H2  Deeper filters absorb the structure: the oracle ceiling at 10^6 is below that at 101.
  H3  Free memory search gains (frozen and refitted) below +0.3% at every depth.
  H4  Charged memory search gains within 3 random s.d. of 0 at every depth.
  H5  Below 0.1% once filters are deep: the mean free (frozen) gain over depths 10^4-10^6 and both ranges < +0.001.
  H6  Buchstab: at depth 10^6, primes / Mertens baseline within 0.003 of e^-gamma / omega(u) (u = ln x0 / ln 10^6:
      2.83 at 10^17, 3.00 at 10^18; predicted 0.9904 and 0.9949).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments import e84_primes as e84
from experiments import prime_data as pd

SEED = 85
BLOCKS = 5000
DEV = 1e14
TESTS = (1e17, 1e18)
DEPTHS = (101, 1000, 10 ** 4, 10 ** 5, 10 ** 6)
CONTROL_DEPTH = 10 ** 4
TOP = e84.TOP
WARM = 100


def charged_gain(count, surv, z, coef, rng, top=TOP, warm=WARM):
    """Sequential search that only knows the counts of blocks it tested. Returns (gain, fraction tested, s.d. of the
    gain of random selections of the same size from the same blocks)."""
    t = len(z)
    windows = e84.WINDOWS
    known = np.zeros(t)
    test = np.zeros(t, bool)
    test[:warm] = True
    known[:warm] = z[:warm]
    scores = []
    for k in range(warm, t):
        x = np.array([known[k - w:k].mean() for w in windows])
        s = float(e84.predict(x[None], coef)[0])
        scores.append(s)
        if s >= np.quantile(scores, 1 - top):
            test[k] = True
            known[k] = z[k]
    sel = test[warm:]
    c, v = count[warm:], surv[warm:]
    return (float(c[sel].sum() / v[sel].sum() / (c.sum() / v.sum()) - 1), float(sel.mean()),
            e84.random_sd(c, v, rng, top=sel.mean()))


def series(p, depth):
    tot, sv, ex = p["count"].sum(1), p[f"surv{depth}"].sum(1), p[f"expected{depth}"].sum(1)
    return tot, sv, ex, e84.z_score(tot, ex)


def measure(tot, sv, ex, coef, rng):
    z = e84.z_score(tot, ex)
    x, y = e84.har(z)
    lead = max(e84.WINDOWS)
    c, s = tot[lead:], sv[lead:]
    refit = e84.crossfit(x, y)
    charged, frac, charged_sd = charged_gain(tot, sv, z, coef, rng)
    return {"calibration": float(tot.sum() / ex.sum()), "tests_per_prime": float(sv.sum() / tot.sum()),
            "var_ratio": float(np.var(z)), "acf1": float(np.corrcoef(z[:-1], z[1:])[0, 1]),
            "gain_frozen": e84.mse_gain(y, e84.predict(x, coef)), "gain_crossfit": e84.mse_gain(y, refit),
            "free": e84.search_gain(c, s, e84.predict(x, coef)), "free_refit": e84.search_gain(c, s, refit),
            "charged": charged, "charged_fraction": frac, "charged_sd": charged_sd,
            "oracle": e84.search_gain(c, s, c / s), "random_sd": e84.random_sd(c, s, rng)}


def simulate(sv, ex, rng, plant=None):
    rate = ex / sv
    if plant:
        phi, sd = plant
        u = np.zeros(len(sv))
        e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), len(sv))
        u[0] = rng.normal(0, sd)
        for t in range(1, len(u)):
            u[t] = phi * u[t - 1] + e[t]
        rate = rate * (1 + u)
    return rng.binomial(sv, np.clip(rate, 0, 1))


def run_range(x, coefs, seed, blocks=BLOCKS):
    rng = np.random.default_rng([SEED, seed])
    p = pd.panel(x, blocks, filters=DEPTHS)
    res = {"x": x, "x0": p["x0"], "blocks": blocks, "primes": int(p["count"].sum()), "depths": {}}
    for d in DEPTHS:
        tot, sv, ex, _ = series(p, d)
        res["depths"][str(d)] = measure(tot, sv, ex, coefs[d], rng)
    tot, sv, ex, _ = series(p, CONTROL_DEPTH)
    for name, plant in CONTROLS:
        res[name] = measure(simulate(sv, ex, rng, plant), sv, ex, coefs[name], rng)
    return res


CONTROLS = (("sieve_random", None), ("planted", e84.PLANT))


def develop(blocks):
    """Coefficients per depth from the development range; for the controls, from one simulated development series
    each (a search that learned the pattern beforehand)."""
    p = pd.panel(DEV, blocks, filters=DEPTHS)
    coefs = {d: e84.fit(*e84.har(series(p, d)[3])) for d in DEPTHS}
    rng = np.random.default_rng([SEED, 99])
    _, sv, ex, _ = series(p, CONTROL_DEPTH)
    for name, plant in CONTROLS:
        coefs[name] = e84.fit(*e84.har(e84.z_score(simulate(sv, ex, rng, plant), ex)))
    dev = run_range(DEV, coefs, 0, blocks)
    dev["coefs"] = {str(d): c.tolist() for d, c in coefs.items()}
    return dev, coefs


def main(out_dir: Path | None = None, quick: bool = False, dev_only: bool = False) -> str:
    out_dir = out_dir or Path("results")
    blocks = 1000 if quick else BLOCKS
    dev, coefs = develop(blocks)
    res = {"dev": dev, "tests": [] if dev_only else [run_range(x, coefs, i + 1, blocks) for i, x in enumerate(TESTS)]}
    text = analyze(res)
    if not dev_only:
        (out_dir / "e85_prime_filters_raw.json").write_text(json.dumps(res, default=float))
        from experiments.e85_figures import make_figures
        make_figures(res, out_dir / "e85_figures", text)
    return text


# ---------------------------------------------------------------- report

def analyze(res) -> str:
    lines = ["Experiment 85: memory on top of every filter, with the information it needs charged", ""]
    for label, r in [("development", res["dev"])] + [("held out", r) for r in res["tests"]]:
        lines += [f"  {label}: x0 = {r['x0']:,} ({r['blocks']} blocks, {r['primes']:,} primes)",
                  "    depth     tests/prime  calib.  var/Poi  acf1     search gain: free    refitted  charged "
                  "(tested)   oracle   random s.d."]
        for d, m in r["depths"].items():
            lines.append(f"    {int(d):<9,d} {m['tests_per_prime']:6.3f}       {m['calibration']:.4f}  "
                         f"{m['var_ratio']:.3f}    {m['acf1']:+.4f}   {m['free']:+.4f}     {m['free_refit']:+.4f}   "
                         f"{m['charged']:+.4f} ({m['charged_fraction']:.2f}, s.d. {m['charged_sd']:.4f})  "
                         f"{m['oracle']:+.4f}  {m['random_sd']:.4f}")
        for name in ("sieve_random", "planted"):
            m = r[name]
            lines.append(f"    {name:12s} (depth {CONTROL_DEPTH:,}) acf1 {m['acf1']:+.4f}  free {m['free']:+.4f}  "
                         f"refitted {m['free_refit']:+.4f} (s.d. {m['random_sd']:.4f})  charged {m['charged']:+.4f} "
                         f"(s.d. {m['charged_sd']:.4f})")
        lines.append("")
    if res["tests"]:
        lines.append("  Pre-registered predictions:")
        for key, label, ok, detail in predictions(res):
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def buchstab(x0, depth):
    """e^-gamma / omega(u): the expected primes / Mertens-baseline ratio, for 2 <= u <= 3."""
    u = np.log(x0) / np.log(depth)
    assert 2 <= u <= 3 + 1e-9
    return float(np.exp(-np.euler_gamma) / ((1 + np.log(u - 1)) / u))


def predictions(res):
    tests = res["tests"]
    keys = [str(d) for d in DEPTHS]
    within = lambda m, k: abs(m[k]) < 3 * m["random_sd"]  # noqa: E731
    out = []
    sr = [t["sieve_random"] for t in tests]
    out.append(("C1", "sieve-random: free, refitted, charged gains within 3 s.d.",
                all(within(m, k) for m in sr for k in ("free", "free_refit"))
                and all(abs(m["charged"]) < 3 * m["charged_sd"] for m in sr),
                "; ".join(f"{m['free']:+.4f}/{m['free_refit']:+.4f} (s.d. {m['random_sd']:.4f}), charged "
                          f"{m['charged']:+.4f} (s.d. {m['charged_sd']:.4f})" for m in sr)))
    pl = [t["planted"] for t in tests]
    out.append(("C2", "planted: refitted and charged gains > 3 s.d.",
                all(m["free_refit"] > 3 * m["random_sd"] and m["charged"] > 3 * m["charged_sd"] for m in pl),
                "; ".join(f"{m['free_refit']:+.4f} (s.d. {m['random_sd']:.4f}), charged {m['charged']:+.4f} "
                          f"(s.d. {m['charged_sd']:.4f})" for m in pl)))
    ok, det = True, []
    for t in tests:
        d = t["depths"]
        steps = [d[a]["tests_per_prime"] / d[b]["tests_per_prime"] - 1 for a, b in zip(keys[:-1], keys[1:])]
        ok &= all(s > d[a]["oracle"] for s, a in zip(steps, keys[:-1]))
        det.append(" ".join(f"{s:+.2f}>{d[a]['oracle']:+.3f}" for s, a in zip(steps, keys[:-1])))
    out.append(("H1", "each tenfold deeper filter beats the oracle ceiling at the shallower depth", ok, "; ".join(det)))
    out.append(("H2", "oracle ceiling at 10^6 below that at 101",
                all(t["depths"][keys[-1]]["oracle"] < t["depths"][keys[0]]["oracle"] for t in tests),
                "; ".join(f"{t['depths'][keys[0]]['oracle']:+.4f} -> {t['depths'][keys[-1]]['oracle']:+.4f}"
                          for t in tests)))
    free = [t["depths"][k][f] for t in tests for k in keys for f in ("free", "free_refit")]
    out.append(("H3", "free memory gains < +0.3% at every depth", max(free) < 0.003, f"max {max(free):+.4f}"))
    ch = [(t["depths"][k]["charged"], t["depths"][k]["charged_sd"]) for t in tests for k in keys]
    out.append(("H4", "charged memory gains within 3 s.d. at every depth", all(abs(g) < 3 * s for g, s in ch),
                f"largest |gain|/s.d. {max(abs(g) / s for g, s in ch):.2f}"))
    deep = [t["depths"][k]["free"] for t in tests for k in keys[2:]]
    out.append(("H5", "mean free gain over depths 10^4-10^6 < +0.1%", float(np.mean(deep)) < 0.001,
                f"mean {np.mean(deep):+.5f} ({', '.join(f'{g:+.4f}' for g in deep)})"))
    cal = [(t["depths"][keys[-1]]["calibration"], buchstab(t["x0"], DEPTHS[-1])) for t in tests]
    out.append(("H6", "Buchstab: calibration at 10^6 within 0.003 of e^-gamma/omega(u)",
                all(abs(c - b) <= 0.003 for c, b in cal), "; ".join(f"{c:.4f} vs {b:.4f}" for c, b in cal)))
    return out


if __name__ == "__main__":
    main(dev_only="--dev" in sys.argv)
