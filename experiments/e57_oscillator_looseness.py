"""Experiment 57: in oscillators, does measured looseness predict and locate openness? (e56, redone properly)

e56's random oscillator ensemble was badly drawn: 19 of 24 groups were fully locked, so the across-group
predictions had nothing to rank. Where groups were open, the information sat in the loosest members (4/4),
and averaging over all 30 members could hide a single loose one (run 8: all-member G 0.000, loosest-3 G
0.079). Post hoc, Spearman(max looseness, G from the loosest 3) was +0.73.

e57 tests that properly, on 30 new groups (30 oscillators each, e55_sims.kuramoto_free, m = 0) drawn from a
range chosen on dynamics alone so that looseness varies: coupling K in [1.0, 2.0], spread of natural
frequencies in [0.3, 1.0], noise in [0.2, 0.5]. A dynamics check of these ranges (other seeds, 8000 steps)
gave 9/30 fully locked groups, 3 partly loose, 18 with 1-30 loose members; no closure numbers were looked at.

Looseness: e56's lambda_i = 1 - |mean_t exp(i phi_i)| on the fitted 70%. Test: e56's (block, ridge,
bootstrap; one time unit), with the measure now fixed in advance as G from the arrangement of the k = 3
loosest members (e54's oscillator features), against the 3 tightest and all 30. Null: the 3 loosest
members of the same parameters' run with another seed. Planted: 5% in r, built from the loosest-3 features.

Predictions, fixed before running (30 groups):
  E0  The test works: planted rise >= 0.025 with lower bound > 0 in >= 24/30, null |G| <= 0.01 (r and
      drift) in >= 24/30.
  E1  Looseness predicts openness: Spearman(max lambda, G_loosest(r)) >= 0.6.
  E2  Looseness locates it: where r is open (G_loosest >= 0.03, lower > 0), G(loosest 3) > G(tightest 3)
      in >= 80%.
  E3  Nobody loose, nothing open: G_loosest(r) <= 0.02 in >= 80% of groups with max lambda < 0.1.
  E4  The phase drift opens with the amount of looseness: Spearman(sum of lambda, G_loosest(drift)) >= 0.6.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e49_school_layer import LAGS
from experiments.e50_school_configuration import _wrap
from experiments.e54_neutral_directions import _lags, _quad, _rates, _spearman
from experiments.e55_free_members import phase_sets
from experiments.e55_sims import kuramoto_free
from experiments.e56_measured_lock import K_SIM, _closure, _sets, looseness

SEED = 57
RUNS = 30
STEPS = 25000
H = 10


def draw(i):
    rng = np.random.default_rng(SEED * 100 + i)
    return {"coupling": float(rng.uniform(1.0, 2.0)), "locked_sd": float(rng.uniform(0.3, 1.0)),
            "noise": float(rng.uniform(0.2, 0.5))}


def _sim(i, seed, ref):
    th = kuramoto_free(m=0, steps=STEPS, seed=seed, **draw(i))
    lam = looseness(th, ref)
    feats = phase_sets(th, ref, _sets(lam, K_SIM))
    zc = np.exp(1j * th).mean(1)
    psi = np.angle(zc)
    z = np.column_stack([np.abs(zc), np.cos(psi), np.sin(psi)])
    block = np.concatenate([_lags(z), _quad(z), _rates(psi)], 1)
    return lam, block, feats, psi, z[:, [0]]


def _run(i):
    rng = np.random.default_rng(SEED * 1000 + i)
    cut = int(0.7 * STEPS)
    ref = np.arange(cut)
    lam, block, feats, psi, r = _sim(i, 1000 + i, ref)
    null = _sim(i, 2000 + i, ref)[2][1]
    rows = np.arange(LAGS + 1, STEPS - H)
    train, test = rows[rows + H < cut], rows[rows >= cut]
    rows = np.concatenate([train, test])
    targets = {"drift": _wrap(psi[rows + H] - psi[rows])[:, None], "r": r[rows + H] - r[rows]}
    extras = {"all": feats[1], "every": feats[0], "tightest": feats[2], "null": null}
    out = _closure(block, extras, targets, train, test, H + LAGS, rng, "r", feats[1][:, 2])
    names = {"all": "loosest", "every": "all", "tightest": "tightest", "null": "null"}
    for t in targets:
        out[t] = {names[k]: v for k, v in out[t].items()}
    return {"i": i, "params": draw(i), "lam": lam.tolist(), "max_lam": float(lam.max()),
            "sum_lam": float(lam.sum()), "mean_r": float(r.mean()), **out}


def predictions(res):
    rs = sorted(res["runs"], key=lambda r: r["i"])
    cal = sum(r["planted"]["after"]["G"] - r["planted"]["before"]["G"] >= 0.025 and r["planted"]["after"]["lo"] > 0
              for r in rs)
    nul = sum(abs(r["r"]["null"]["G"]) <= 0.01 and abs(r["drift"]["null"]["G"]) <= 0.01 for r in rs)
    g = [r["r"]["loosest"]["G"] for r in rs]
    rho1 = _spearman([r["max_lam"] for r in rs], g)
    op = [r for r in rs if r["r"]["loosest"]["G"] >= 0.03 and r["r"]["loosest"]["lo"] > 0]
    win = sum(r["r"]["loosest"]["G"] > r["r"]["tightest"]["G"] for r in op)
    tight = [r for r in rs if r["max_lam"] < 0.1]
    closed = sum(r["r"]["loosest"]["G"] <= 0.02 for r in tight)
    rho4 = _spearman([r["sum_lam"] for r in rs], [r["drift"]["loosest"]["G"] for r in rs])
    return [
        ("E0", "the test works (planted >= 24/30, null >= 24/30)", cal >= 24 and nul >= 24,
         f"planted {cal}/30, null {nul}/30"),
        ("E1", "looseness predicts openness: Spearman(max lambda, G_loosest(r)) >= 0.6", rho1 >= 0.6, f"rho {rho1:+.2f}"),
        ("E2", "looseness locates it: G(loosest) > G(tightest) where r is open (>= 80%)",
         bool(op) and win >= 0.8 * len(op), f"{win}/{len(op)}"),
        ("E3", "nobody loose, nothing open: G_loosest(r) <= 0.02 where max lambda < 0.1 (>= 80%)",
         bool(tight) and closed >= 0.8 * len(tight), f"{closed}/{len(tight)}"),
        ("E4", "drift opens with the amount of looseness: Spearman(sum lambda, G_loosest(drift)) >= 0.6",
         rho4 >= 0.6, f"rho {rho4:+.2f}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 57: in oscillators, does measured looseness predict and locate openness?", ""]
    for r in sorted(res["runs"], key=lambda r: r["max_lam"]):
        c = lambda t: "  ".join(f"{k} {r[t][k]['G']:+.3f}" for k in ("loosest", "tightest", "all", "null"))  # noqa: E731
        lines.append(f"  group {r['i']:2d}  max lambda {r['max_lam']:.2f} sum {r['sum_lam']:5.1f} mean r "
                     f"{r['mean_r']:.2f} | r: {c('r')} [lo {r['r']['loosest']['lo']:+.3f}] | drift: {c('drift')}"
                     f" | planted {r['planted']['before']['G']:+.3f} -> {r['planted']['after']['G']:+.3f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    jobs = list(range(4 if quick else RUNS))
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run, jobs))
    res = {"runs": runs}
    (out_dir / "e57_oscillator_looseness_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e57_figures import make_figures
    make_figures(res, out_dir / "e57_figures", text)
    return text


if __name__ == "__main__":
    main()
