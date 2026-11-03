"""Experiment 40: e39 with a tank-centre term in each fish's own behaviour.

e39 diagnosis. On 8-fish virtual groups (time-shifted, never interacting)
stage 1 eliminated "none" in 6 of 6 and kept topological attraction (k3-C,
k6-C, ...), and stage 0 fired in 3 of 6. A fish's "attraction" to other
fish's time-shifted positions points roughly towards the tank centre, so the
pairwise terms were standing in for an own preference for the centre that
the twin lacked (wall terms act only within about 15 cm). The real-data
eliminations on 8 fish are therefore not trustworthy.

Single change, everything else as in e39 (same candidates, stages,
thresholds, seeds, synthetic truths refitted with the new own terms):
own terms gain c_centre * sin(bearing to the tank centre - theta) * d_c / 50 cm.

Predictions, fixed before running:
  E1  Virtual groups: stage 0 keeps F_anti and "none" survives stage 1 in
      >= 7 of 8 (e39: 2 of 8).
  E2  Synthetic: the truth survives stages 0-2, and the answer contains it
      with <= 3 candidates, in all of T0-T4 (e39's A1 and A4 still hold).
  E3  Real schools: stage 0 eliminates F_anti in >= 6 of 8, and in no
      virtual group (with E1, this makes the e39 B1 result interpretable).
  E4  Mechanism: the fitted centre coefficient (twin fit) is > 0 in all 8
      real recordings and all 8 virtual groups.
  E5  Spacing is not fixed by this change: the real-data answer is empty
      (no candidate passes stage 2) in >= 6 of 8 recordings.
All e39 statistics (pruning, costs, micro selection, T5) are reported again.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiments.e39_multiscale_elimination as e39
from experiments.e31_real_fish import TRAIN, kinematics, virtual
from experiments.e32_fish_closed_twin import turns
from experiments.fish_data import RECORDINGS, load


def _centre_on():
    e39.CENTRE, e39.N_OWN = True, 6


def centre_coef(p, th, sp):
    l1, l2, ok_in = turns(th, sp)
    x = e39.own_features(p, th, l1, l2)[:-1]
    ok = ok_in[1:].copy()
    ok[int(TRAIN * len(th)):] = False
    return float(np.linalg.lstsq(x[ok], l1[1:][ok], rcond=None)[0][5])


def _synthetic(name):
    _centre_on()
    return e39._synthetic(name)


def _real(args):
    _centre_on()
    n, rec, kind = args
    out = e39._real(args)
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    out["centre_coef"] = centre_coef(p, th, np.where(sp > 100.0, 0.0, sp))
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    tasks = [(n, r, k) for n, recs in RECORDINGS.items() for r in recs for k in ("real", "virtual")]
    print(f"  {len(e39.TRUTHS)} synthetic truths + {len(tasks)} real/virtual recordings ...",
          flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 2))) as pool:
        synth = pool.map(_synthetic, e39.TRUTHS)
        real = pool.map(_real, tasks)
        rows = {"synthetic": list(synth), "real": list(real)}
    (out_dir / "e40_elimination_centre_twin_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    with contextlib.redirect_stdout(io.StringIO()):
        base = e39.analyze(rows)
    base = base.replace("Experiment 39: multi-scale elimination of candidate interaction rules",
                                     "Experiment 40: e39 with a tank-centre own term")
    syn = {r["name"]: r for r in rows["synthetic"]}
    real = [r for r in rows["real"] if r["kind"] == "real"]
    virt = [r for r in rows["real"] if r["kind"] == "virtual"]
    in_class = [syn[t] for t in ("T0", "T1", "T2", "T3", "T4")]
    e1n = sum(r["z0"] <= e39.Z0 and "none" in r["surv1"] for r in virt)
    e2 = all(r["truth"] in r["surv2"] and r["truth"] in r["answer"] and len(r["answer"]) <= 3
             for r in in_class)
    e3n = sum(r["z0"] > e39.Z0 for r in real)
    e3v = sum(r["z0"] > e39.Z0 for r in virt)
    e4 = all(r["centre_coef"] > 0 for r in real + virt)
    e5n = sum(not r["answer"] for r in real)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines = [base, "", "  Centre coefficient (twin fit), real / virtual:"]
    for r, v in zip(real, virt):
        lines.append(f"  {r['N']:3d}{r['rec']:4d}{r['centre_coef']:10.4f}{v['centre_coef']:10.4f}")
    lines += ["", "  e40 pre-registered predictions (the A/B lines above are e39's, reported again):",
              f"  {tag(e1n >= 7)}  E1 virtual: F_anti kept and 'none' survives stage 1 in >= 7 of 8 [{e1n}/8]",
              f"  {tag(e2)}  E2 synthetic: truth survives and answer contains it (<= 3), T0-T4",
              f"  {tag(e3n >= 6 and e3v == 0)}  E3 real: stage 0 fires in >= 6 of 8 [{e3n}/8] "
              f"and in no virtual group [{e3v}/8]",
              f"  {tag(e4)}  E4 centre coefficient > 0 in all real and virtual groups",
              f"  {tag(e5n >= 6)}  E5 real answer empty (spacing not fixed) in >= 6 of 8 [{e5n}/8]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
