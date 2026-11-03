"""Experiment 36: pigeon flocks again, with a coarser micro step and a gated
positive control.

e35 failed at identifying the twin: at the 0.2 s GPS step the turn model
mostly learned the tracks' smoothing filter (own-lag coefficients 1.9, -1.5,
0.54), the fitted interactions were tiny, homing was too weak (loners drift
about 1 km), and the fitted model did not flock when simulated (spread
110-200 m against 6-8 m real). Its synthetic positive control therefore had
nothing to detect, and no prediction required it to.

Changes from e35 (everything else identical: same flocks and flights, all-
birds-flying samples, homing polynomial, 4-nearest-neighbour alignment /
attraction / repulsion, virtual flocks by time shift, S_proj with 64
replicas, 70/30 split):

  micro step 1 s (every 10th 10 Hz row), well beyond the 0.4 s smoothing;
  one lag of the bird's own turn;
  macro step 2 s (2 micro steps).

Predictions, fixed before running (each for flocks A, B and C):

  G   Gate (positive control): the synthetic flock simulated from the full
      fitted model must hold together, with mean distance to the centroid at
      most 3x the real flock's, and must show interaction, S_proj >= 0.2.
      A flock that fails the gate has its R3-R5 reported but not counted:
      the twin is not identified there.
  R1  Synthetic loners: |S_proj| <= 0.03.
  R2  Synthetic flock: S_proj from the fitted twin within 0.05 of the true twin's.
  R3  Virtual flocks: S_proj <= 0.1.
  R4  Real flocks: S_proj >= S_proj(virtual) + 0.2.
  R5  State dependence: fitting G0_proj on test instead of training states
      changes real S_proj by <= 0.05.
  R6  Smoothing no longer dominates: |residual lag-1 autocorrelation| <= 0.2.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiments.e35_pigeons as e35
from experiments.e31_real_fish import social_terms
from experiments.pigeon_data import FLOCKS, flights, load

CONFIG = dict(dt=1.0, tau=2, lags=1, stride=10)


def spread(p):
    c = p - p.mean(1, keepdims=True)
    return float(np.hypot(c[..., 0], c[..., 1]).mean())


def residual_autocorr(series, coef):
    num = []
    for s in series:
        own = np.concatenate([np.ones(s["theta"].shape + (1,)), e35.homing(s["p"], s["theta"]),
                              s["lags"]], -1)
        x = np.concatenate([own, social_terms(s["p"], s["theta"])], -1)
        r = s["target"] - x @ coef
        a, b = r[:-1], r[1:]
        ok = np.isfinite(a) & np.isfinite(b)
        num.append((a[ok], b[ok]))
    a = np.concatenate([u for u, _ in num])
    b = np.concatenate([v for _, v in num])
    return float(np.corrcoef(a, b)[0, 1])


def _flock(flock):
    e35.configure(**CONFIG)
    rng = np.random.default_rng(36_000 + ord(flock))
    real = [load(f, CONFIG["stride"]) for f in flights(flock)]
    out = {"flock": flock, "minutes": sum(len(s["cont"]) for s in real) * CONFIG["dt"] / 60,
           "spread_real": float(np.mean([spread(s["p"]) for s in real]))}
    out["real"] = e35.analyse(real, rng)
    out["virtual"] = e35.analyse([e35.virtual(s) for s in real], rng)
    series = [e35.prepare(s) for s in real]
    for s in series:
        s["train_stop"] = len(s["theta"])
    coef, res = e35.fit(series)
    out["coef_all"] = coef.tolist()
    out["res_autocorr"] = residual_autocorr(series, coef)
    speed = np.median(np.concatenate([s["speed"] for s in real]), 0)
    steps = sum(len(s["cont"]) for s in real)
    for kind, social in (("flock", True), ("loners", False)):
        sim = e35.simulate(coef, res, speed, real[0]["p"][0], real[0]["theta"][0], steps, social, rng)
        out["synthetic_" + kind] = e35.analyse([sim], rng, true_twin=(coef[:e35.N_OWN], res)) \
            | {"spread": spread(sim["p"])}
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    for fl in FLOCKS:
        for f in flights(fl):
            load(f, CONFIG["stride"])
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(_flock, FLOCKS))
    (out_dir / "e36_pigeons_coarse_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    n_own = 4 + CONFIG["lags"]
    lines = ["Experiment 36: pigeon flocks, 1 s micro step, gated positive control", "",
             f"  {'flock':>5}{'min':>6}{'spread real/synth':>19}{'synth S':>9}{'gate':>6}"
             f"{'S_proj':>8}{'virtual':>9}{'alt':>7}{'loners':>8}{'true':>7}{'res ac':>8}"]
    gate = {}
    for r in rows:
        sf = r["synthetic_flock"]
        gate[r["flock"]] = sf["spread"] <= 3 * r["spread_real"] and sf["S_proj"] >= 0.2
        lines.append(f"  {r['flock']:>5}{r['minutes']:6.1f}{r['spread_real']:9.1f}/{sf['spread']:<9.1f}"
                     f"{sf['S_proj']:9.3f}{'pass' if gate[r['flock']] else 'FAIL':>6}"
                     f"{r['real']['S_proj']:8.3f}{r['virtual']['S_proj']:9.3f}{r['real']['S_proj_alt']:7.3f}"
                     f"{r['synthetic_loners']['S_proj']:8.3f}{sf['true_twin']['S_proj']:7.3f}"
                     f"{r['res_autocorr']:8.3f}")
        c = r["coef_all"]
        lines.append(f"        coefficients: bias {c[0]:.3f}, homing {c[1]:.3f} {c[2]:.3f} {c[3]:.3f}, "
                     f"own lag {c[4]:.3f}, align {c[n_own]:.3f}, attract {c[n_own + 1]:.3f}, "
                     f"repel {c[n_own + 2]:.3f}")
    ok = [r for r in rows if gate[r["flock"]]]
    tag = lambda v: "PASS" if v else "FAIL"
    counted = lambda f: tag(all(f(r) for r in ok)) if ok else "n/a "
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(all(gate.values()))}  G  gate: synthetic flock coheres and S_proj >= 0.2 "
              f"[passed: {', '.join(k for k, v in gate.items() if v) or 'none'}]",
              f"  {tag(all(abs(r['synthetic_loners']['S_proj']) <= 0.03 for r in rows))}  "
              f"R1 synthetic loners |S_proj| <= 0.03",
              f"  {tag(all(abs(r['synthetic_flock']['S_proj'] - r['synthetic_flock']['true_twin']['S_proj']) <= 0.05 for r in rows))}  "
              f"R2 synthetic flock: fitted twin within 0.05 of true twin",
              f"  {counted(lambda r: r['virtual']['S_proj'] <= 0.1)}  R3 virtual flocks S_proj <= 0.1 (gated flocks)",
              f"  {counted(lambda r: r['real']['S_proj'] >= r['virtual']['S_proj'] + 0.2)}  "
              f"R4 real >= virtual + 0.2 (gated flocks)",
              f"  {counted(lambda r: abs(r['real']['S_proj'] - r['real']['S_proj_alt']) <= 0.05)}  "
              f"R5 state dependence <= 0.05 (gated flocks)",
              f"  {tag(all(abs(r['res_autocorr']) <= 0.2 for r in rows))}  R6 |residual autocorrelation| <= 0.2"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
