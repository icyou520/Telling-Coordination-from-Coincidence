"""Experiment 87: RANDU. Can the interaction share catch a flawed random-number generator?

RANDU (IBM, 1960s: x_{n+1} = 65539 x_n mod 2^31) passed the tests of its day, but every three consecutive outputs
satisfy x_{n+2} = 6 x_{n+1} - 9 x_n (mod 2^31): in three dimensions its points lie on 15 planes. The flaw is
arithmetic modulo 2^31, which leaves almost no linear correlation. The share is built from linear forecasts, so it
should be blind to it. That is the blind spot under test.

Members: K = 8 interleaved streams (member i at time t is output tK + i), the group their sum; T = 20,000 steps.
Measured with twinshare (cross-fitted share with its checks, and the memory gap), on raw outputs u in [0, 1).
Generators:
    pcg64      numpy's default (good);
    minstd     Park-Miller x_{n+1} = 16807 x_n mod (2^31 - 1) (adequate for this);
    randu      as above;
    leaky      a positive control with a cross-stream flaw a share should see: pcg64, but with probability 0.1 an
               output repeats the previous step's output of the next stream (member i at t takes member i+1's value
               at t-1, a buffer bug).
Classical check: 1,000,000 non-overlapping triples binned in 20^3 cells, chi-square against uniform.

Development (seeds 1, 2):
  * pcg64, minstd, randu: S -0.0004 to +0.0000, memory gap <= 0.0002, "No coordination detected";
  * leaky: S +0.0088 and +0.0072, memory gap 0.0093 and 0.0076, "Coordination detected (small)";
  * triple test: randu rejected beyond floating point (p underflows to 0); pcg64, minstd and leaky p 0.04-0.5.
  So each tool sees what the other misses: the share sees dependence between streams across time; the triple
  test sees RANDU's static modular structure.

Pre-registered predictions (seeds 3-7):
  N1  Good generators (pcg64, minstd): "No coordination detected" and triple test p > 0.001, every seed.
  N2  The share is blind to RANDU: "No coordination detected", |S| < 0.002 and memory gap < 0.002, every seed.
  N3  The triple test catches RANDU: p < 10^-10, every seed.
  N4  The share catches the leak: "Coordination detected" with S > 0.004, every seed.
  N5  The triple test misses the leak: p > 0.001, every seed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twinshare" / "src"))
from twinshare import analysis, data  # noqa: E402

SEED = 87
K = 8
STEPS = 20000
LEAK = 0.1
TRIPLES = 1_000_000
BINS = 20
GENERATORS = ("pcg64", "minstd", "randu", "leaky")


def lcg(seed, n, a, m):
    x = seed % m or 1
    out = np.empty(n)
    for i in range(n):
        x = a * x % m
        out[i] = x / m
    return out


def outputs(name, n, seed):
    rng = np.random.default_rng([SEED, seed])
    if name == "pcg64":
        return rng.random(n)
    if name == "minstd":
        return lcg(int(rng.integers(1, 2 ** 31 - 1)), n, 16807, 2 ** 31 - 1)
    if name == "randu":
        return lcg(int(rng.integers(0, 2 ** 30)) * 2 + 1, n, 65539, 2 ** 31)
    if name == "leaky":
        u = rng.random(n).reshape(-1, K)
        hit = rng.random(u.shape) < LEAK
        for t in range(1, len(u)):
            u[t, hit[t]] = np.roll(u[t - 1], -1)[hit[t]]
        return u.ravel()
    raise ValueError(name)


def triple_test(u):
    v = u[: 3 * (len(u) // 3)].reshape(-1, 3)
    idx = np.minimum((v * BINS).astype(int), BINS - 1)
    counts = np.bincount((idx[:, 0] * BINS + idx[:, 1]) * BINS + idx[:, 2], minlength=BINS ** 3)
    e = len(v) / BINS ** 3
    stat = float(((counts - e) ** 2 / e).sum())
    return stat, float(chi2.sf(stat, BINS ** 3 - 1)), float(chi2.logsf(stat, BINS ** 3 - 1) / np.log(10))


def run(name, seed, steps=STEPS, triples=TRIPLES):
    x = outputs(name, steps * K, seed).reshape(steps, K)
    r = analysis.analyze(data.from_arrays(x, members=[f"stream {i}" for i in range(K)]), seed=seed, detail=False)
    stat, p, log10p = triple_test(outputs(name, 3 * triples, seed + 1000))
    return {"gen": name, "seed": seed, "S": r.S, "interval": list(r.interval), "memory_gap": r.memory_gap["S_pred"],
            "verdict": r.verdict, "null_bar": analysis.null_bar(r),
            "lag1": float(np.corrcoef(x[:-1].ravel(), x[1:].ravel())[0, 1]),
            "chi2": stat, "p": p, "log10p": log10p}


def describe(rows):
    lines = ["    generator  seed   S         90% interval          memory gap  null bar   verdict"
             "                      triple test log10 p"]
    for r in rows:
        lines.append(f"    {r['gen']:9s}  {r['seed']:<5d} {r['S']:+.4f}  [{r['interval'][0]:+.4f}, {r['interval'][1]:+.4f}]  "
                     f"{r['memory_gap']:+.4f}     {r['null_bar']:+.4f}    {r['verdict']:28s} {r['log10p']:9.1f}")
    return lines


SEEDS = (3, 4, 5, 6, 7)
NONE = "No coordination detected"


def predictions(rows):
    by = {g: [r for r in rows if r["gen"] == g] for g in GENERATORS}
    p = lambda rs: ", ".join(f"{r['log10p']:.1f}" for r in rs)  # noqa: E731
    good = by["pcg64"] + by["minstd"]
    out = [("N1", "good generators: no coordination, triple p > 0.001",
            all(r["verdict"] == NONE and r["p"] > 1e-3 for r in good),
            f"verdicts {sum(r['verdict'] == NONE for r in good)}/{len(good)}; log10 p {p(good)}")]
    ra = by["randu"]
    out.append(("N2", "share blind to RANDU: no coordination, |S| and memory gap < 0.002",
                all(r["verdict"] == NONE and abs(r["S"]) < 0.002 and abs(r["memory_gap"]) < 0.002 for r in ra),
                "S " + ", ".join(f"{r['S']:+.4f}" for r in ra)))
    out.append(("N3", "triple test catches RANDU: p < 1e-10", all(r["log10p"] < -10 for r in ra), f"log10 p {p(ra)}"))
    lk = by["leaky"]
    out.append(("N4", "share catches the leak: detected, S > 0.004",
                all(r["verdict"].startswith("Coordination detected") and r["S"] > 0.004 for r in lk),
                "S " + ", ".join(f"{r['S']:+.4f}" for r in lk)))
    out.append(("N5", "triple test misses the leak: p > 0.001", all(r["p"] > 1e-3 for r in lk), f"log10 p {p(lk)}"))
    return out


def analyze(rows):
    lines = ["Experiment 87: RANDU. Can the interaction share catch a flawed random-number generator?", ""]
    lines += describe(rows) + ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(rows):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    seeds, kw = ((101,), dict(steps=4000, triples=200_000)) if quick else (SEEDS, {})
    rows = [run(g, s, **kw) for s in seeds for g in GENERATORS]
    (out_dir / "e87_randu_raw.json").write_text(json.dumps(rows))
    text = analyze(rows)
    from experiments.e87_figures import make_figures
    make_figures(rows, out_dir / "e87_figures", text)
    return text


if __name__ == "__main__":
    main()
