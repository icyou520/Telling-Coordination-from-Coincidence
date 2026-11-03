"""Experiment 33: a closure-robust interaction share on real fish schools.

e32 fixed the twin (persistent turning) but no low-dimensional macro state
closed it, mostly for 8 fish: the micro-informed twin prediction g0(x) uses
every fish's heading and so beats any law in polarization phi even for
fish that do not interact (S on synthetic loners about -0.1).

The comparator is changed to like-for-like. The twin's law in phi is the
projection of its micro-informed prediction onto phi, over the observed
states:

    G0_proj(phi) = cubic fit of g0(x_t) against phi_t over training states,
    S_proj       = 1 - MSE(G) / MSE(G0_proj)   on the test states,

with G the cubic fit of the real phi(t + 1 s) against phi(t), as before.
For a group without interaction, the real law and G0_proj both estimate
E[phi' | phi] under the same state distribution, so S_proj = 0 whether or
not the twin is closed. The price is that G0_proj depends on which states
it is fitted on (the e30 ambiguity); C7 tests how much.

Twin, data, splits, virtual groups and synthetic groups are those of e32
(macro state M1 = phi; 64 replicas; at most 1500 train and 1500 test states).

Predictions, fixed before running:

  C1  Synthetic loners: |S_proj| <= 0.03 for N = 8 and 39.
  C2  Synthetic school: S_proj from the fitted twin within 0.05 of the
      true twin's.
  C3  Synthetic school: S_proj >= S_proj(loners) + 0.2.
  C4  Real virtual groups: S_proj <= 0.1 in every recording.
  C5  Real schools: S_proj >= S_proj(virtual) + 0.2 in every recording.
  C6  e32's values were conservative: S_proj >= S_micro (the e32
      comparator) in at least 7 of 8 real recordings.
  C7  State dependence: fitting G0_proj on test instead of training states
      changes S_proj by <= 0.05 in every real recording and synthetic group.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import TAU, TRAIN, kinematics, polarization, virtual
from experiments.e32_fish_closed_twin import (N_OWN, fit_model, simulate_synthetic,
                                              turns, twin_predict)
from experiments.fish_data import RECORDINGS, load

MAX_STATES = 1500


def _sub(idx):
    return idx if len(idx) <= MAX_STATES else idx[np.linspace(0, len(idx) - 1, MAX_STATES).astype(int)]


def shares(states, c, res, rng) -> dict:
    phi = polarization(states[1])
    x, z = phi[:-1], phi[1:]
    cut = int(TRAIN * len(x))
    tr, te = _sub(np.arange(cut)), _sub(np.arange(cut, len(x)))
    g0_tr = twin_predict(tuple(a[tr] for a in states), c, res, rng)
    g0_te = twin_predict(tuple(a[te] for a in states), c, res, rng)
    g = np.polyval(np.polyfit(x[:cut], z[:cut], 3), x[te])
    proj = np.polyval(np.polyfit(x[tr], g0_tr, 3), x[te])
    proj_alt = np.polyval(np.polyfit(x[te], g0_te, 3), x[te])
    mse = lambda pred: np.mean((z[te] - pred) ** 2)
    return {"S_proj": float(1 - mse(g) / mse(proj)),
            "S_proj_alt": float(1 - mse(g) / mse(proj_alt)),
            "S_micro": float(1 - mse(g) / mse(g0_te))}


def analyse(p, theta, speed, rng, true_twin=None):
    coef, res, _ = fit_model(p, theta, speed, int(TRAIN * len(theta)))
    l1, l2, _ = turns(theta, speed)
    m = np.arange(0, len(theta), TAU)
    states = (p[m], theta[m], speed[m], l1[m], l2[m])
    out = {"coef": coef.tolist(), **shares(states, coef[:N_OWN], res, rng)}
    if true_twin is not None:
        out["true_twin"] = shares(states, *true_twin, rng)
    return out


def _real(args):
    n, rec = args
    rng = np.random.default_rng(33_000 + 10 * n + rec)
    pos = load(n, rec)
    out = {"n": n, "rec": rec}
    for kind, data in (("real", pos), ("virtual", virtual(pos))):
        p, th, sp = kinematics(data)
        out[kind] = analyse(p, th, np.where(sp > 100.0, 0.0, sp), rng)
    return out


def _synthetic(n):
    rng = np.random.default_rng(33_500 + n)
    p, th, sp = kinematics(load(n, 0))
    sp = np.where(sp > 100.0, 0.0, sp)
    coef, res, _ = fit_model(p, th, sp, int(TRAIN * len(th)))
    speed = np.median(sp, axis=0)
    out = {"n": n}
    for kind, social in (("school", True), ("loners", False)):
        ps, ths = simulate_synthetic(coef, res, speed, p[0], th[0], social, rng)
        spd = np.broadcast_to(speed, ths.shape).copy()
        out[kind] = analyse(ps, ths, spd, rng, true_twin=(coef[:N_OWN], res))
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    tasks = [(n, r) for n, recs in RECORDINGS.items() for r in recs]
    print(f"  {len(tasks)} recordings + 2 synthetic groups ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(10, (os.cpu_count() or 2) - 2))) as pool:
        synth = pool.map(_synthetic, (8, 39))
        real = pool.map(_real, tasks)
        rows = {"real": list(real), "synthetic": list(synth)}
    (out_dir / "e33_fish_projected_twin_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    real, synth = rows["real"], rows["synthetic"]
    lines = ["Experiment 33: closure-robust interaction share on fish schools", "",
             "  Part A, synthetic fish (known answer):",
             f"  {'N':>3}{'group':>8}{'S_proj':>8}{'true twin':>11}{'S_micro':>9}{'alt states':>12}"]
    for s in synth:
        for kind in ("school", "loners"):
            d = s[kind]
            lines.append(f"  {s['n']:3d}{kind:>8}{d['S_proj']:8.3f}{d['true_twin']['S_proj']:11.3f}"
                         f"{d['S_micro']:9.3f}{d['S_proj_alt']:12.3f}")
    lines += ["", "  Part B, real recordings:",
              f"  {'N':>3}{'rec':>4}{'S_proj':>8}{'virtual':>9}{'S_micro':>9}{'alt states':>12}"
              f"{'virt. micro':>13}"]
    for r in real:
        re, vi = r["real"], r["virtual"]
        lines.append(f"  {r['n']:3d}{r['rec']:4d}{re['S_proj']:8.3f}{vi['S_proj']:9.3f}"
                     f"{re['S_micro']:9.3f}{re['S_proj_alt']:12.3f}{vi['S_micro']:13.3f}")
    c1 = all(abs(s["loners"]["S_proj"]) <= 0.03 for s in synth)
    c2 = all(abs(s["school"]["S_proj"] - s["school"]["true_twin"]["S_proj"]) <= 0.05 for s in synth)
    c3 = all(s["school"]["S_proj"] >= s["loners"]["S_proj"] + 0.2 for s in synth)
    c4 = all(r["virtual"]["S_proj"] <= 0.1 for r in real)
    c5 = all(r["real"]["S_proj"] >= r["virtual"]["S_proj"] + 0.2 for r in real)
    n6 = sum(r["real"]["S_proj"] >= r["real"]["S_micro"] for r in real)
    alt = [abs(r["real"]["S_proj"] - r["real"]["S_proj_alt"]) for r in real] + \
          [abs(s[k]["S_proj"] - s[k]["S_proj_alt"]) for s in synth for k in ("school", "loners")]
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(c1)}  C1 synthetic loners |S_proj| <= 0.03 "
              f"[max {max(abs(s['loners']['S_proj']) for s in synth):.3f}]",
              f"  {tag(c2)}  C2 synthetic school: fitted twin within 0.05 of true twin",
              f"  {tag(c3)}  C3 synthetic school S_proj >= loners + 0.2",
              f"  {tag(c4)}  C4 virtual groups S_proj <= 0.1 [max {max(r['virtual']['S_proj'] for r in real):.3f}]",
              f"  {tag(c5)}  C5 real S_proj >= virtual + 0.2 [min gap "
              f"{min(r['real']['S_proj'] - r['virtual']['S_proj'] for r in real):.3f}]",
              f"  {tag(n6 >= 7)}  C6 S_proj >= S_micro in >= 7 of 8 recordings [{n6}/8]",
              f"  {tag(max(alt) <= 0.05)}  C7 state dependence <= 0.05 [max {max(alt):.3f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
