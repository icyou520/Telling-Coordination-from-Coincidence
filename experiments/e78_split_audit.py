"""e78: does any fish-school verdict depend on testing the last 30% of each recording?

e77 found that a chronological 70/30 split moved the zebrafish interaction share from 0.54 to 0.04 in one
group: the end of a drifting recording is not like its start. The closure and influence experiments
e51-e60 all tested on the last 30% of each recording, and fitted every reference stage (scaling, place maps,
looseness, the own-rule innovation model) on the first 70%. This audit asks whether their real-data
verdicts depend on that choice.

Segment rotation. Each experiment is rerun with its own code and scored with its own pre-registered
predictions(), with the held-out 30% moved:
  last    rows in [0.7 T, T)      (the published choice)
  first   rows in [0, 0.3 T)
  middle  rows in [0.35 T, 0.65 T)
Training rows are those whose target ends before the test segment (row + gap < A, the original rule), or
whose history starts after it with the same gap plus the lag window (row >= B + gap + LAGS). Every
reference stage that used the first 70% uses all rows outside the test segment instead. The hashed modules
are not edited: in each worker process the real-data functions are recompiled from their own source with
the 70% expressions replaced (asserted, so a mismatch fails loudly), and the shared _rows helper is
replaced. Simulations keep their original split; they are stationary and their verdicts must not move.

Experiments audited: e51 (closure of shape, motion and turn), e52 (influence map), e53 (zebrafish and
pigeons), e56 (looseness), e58 (innovation), e59 (slaving), e60 (slow features). Not audited, with reasons:
e48-e50 (superseded by e51's trusted test); e61-e63 (interleaved trial blocks, already not chronological);
e64-e65 (superseded by e71's no-fitting test); e66 onward (cross-fitted, audited in e76 and e77).

Real-data predictions per experiment (24): e51 K3-K6; e52 D1-D4; e53 Z1-Z4, P1-P4; e56 C1-C2; e58 F1-F4;
e59 T3; e60 U3.

Pre-registered predictions:
  R0 machinery: with the last segment, the rotated pipelines reproduce every published raw result
     (largest absolute difference over all numbers <= 1e-9, all seven experiments).
  R1 simulations unchanged: every simulation-only prediction has its published verdict in both rotations.
  H1 the next turn is open: e51 K3 passes in both rotations.
  H2 shape is closed: G(shape) < 0.05 in >= 5/6 real schools, both rotations.
  H3 motion is closed: G(motion) < 0.03 in >= 5/6 real schools, both rotations.
  H4 the front steers: e52 D3 and e53 Z3 pass in both rotations.
  H5 the learned map: e52 D1 and D4 pass in both rotations.
  H6 schools more self-contained than their virtual twins: e51 K6 passes in both rotations.
  H7 pigeons closed at 1 s: e53 P1 fails in both rotations.
  H8 size: median e51 G(turn) over the six real schools lies in [0.06, 0.25] in both rotations
     (published 0.13).
  V1 verdicts: >= 20 of the 24 real-data verdicts match the published ones, in each rotation.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEGMENTS = {"last": (0.7, 1.0), "first": (0.0, 0.3), "middle": (0.35, 0.65)}
MODULES = {
    "e51": ("e51_trusted_closure", ["_run"]),
    "e52": ("e52_influence_map", ["_run"]),
    "e53": ("e53_influence_replication", ["_run"]),
    "e56": ("e56_measured_lock", ["_run_fish"]),
    "e58": ("e58_innovation", ["_run_fish"]),
    "e59": ("e59_slaving", ["_run"]),
    "e60": ("e60_slow_features", ["_data"]),
}
ROWS_USERS = ("e53_influence_replication", "e56_measured_lock", "e58_innovation", "e59_slaving",
              "e60_slow_features")
REAL = {"e51": ["K3", "K4", "K5", "K6"], "e52": ["D1", "D2", "D3", "D4"],
        "e53": ["Z1", "Z2", "Z3", "Z4", "P1", "P2", "P3", "P4"], "e56": ["C1", "C2"],
        "e58": ["F1", "F2", "F3", "F4"], "e59": ["T3"], "e60": ["U3"]}
SIM = {"e51": ["K1", "K2"], "e52": ["D0"], "e56": ["S0", "A1", "A2", "A3", "B1", "B2"],
       "e58": ["V1", "S0", "I1", "I2", "I3"], "e59": ["T1", "T2"], "e60": ["U1", "U2"]}
REF = re.compile(r"np\.arange\(int\(0\.7 \* (T|len\(th\)|len\(tho\))\)\)")
SPLIT = "rows[rows + hmax < cut], rows[rows >= cut]"
SEGMENT = None
_INSTALLED = None


def bounds(seg, T):
    a, b = seg
    return int(a * T), int(b * T)


def make_helpers(seg):
    from experiments.e49_school_layer import LAGS

    def ref(T):
        A, B = bounds(seg, T)
        return np.setdiff1d(np.arange(T), np.arange(A, B))

    def split(rows, T, g):
        A, B = bounds(seg, T)
        test = rows[(rows >= A) & (rows < B)]
        train = rows[(rows + g < A) | (rows >= B + g + LAGS)]
        return train, test

    def _rows(brk, T, h):
        bad = np.zeros(T, bool)
        for s in np.nonzero(brk)[0]:
            bad[max(0, s - h - 1):s + LAGS + 2] = True
        rows = np.arange(LAGS + 1, T - h)
        rows = rows[~bad[rows]]
        return split(rows, T, h)

    return ref, split, _rows


def install(seg):
    """Recompile the real-data functions with the test segment moved; returns the replacement counts."""
    global _INSTALLED
    if _INSTALLED == seg:
        return
    ref, split, rows_fn = make_helpers(seg)
    counts = {}
    for key, (name, funcs) in MODULES.items():
        mod = importlib.import_module("experiments." + name)
        mod._AUDIT_REF, mod._AUDIT_SPLIT = ref, split
        for f in funcs:
            src = inspect.getsource(getattr(mod, f))
            src, n_ref = REF.subn(r"_AUDIT_REF(\1)", src)
            n_split = src.count(SPLIT)
            src = src.replace(SPLIT, "_AUDIT_SPLIT(rows, T, hmax)")
            assert n_ref + n_split > 0, (name, f)
            counts[f"{key}.{f}"] = (n_ref, n_split)
            exec(compile(src, inspect.getsourcefile(mod), "exec"), mod.__dict__)
    for name in ROWS_USERS:
        importlib.import_module("experiments." + name)._rows = rows_fn
    _INSTALLED = seg
    return counts


def _call(job):
    seg, modname, fname, args = job
    install(seg)
    return getattr(importlib.import_module(modname), fname)(*args)


class AuditPool:
    """Stands in for ProcessPoolExecutor inside each module's main(): same map, workers patched first."""

    def __init__(self, max_workers=None):
        self.pool = ProcessPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown()

    def map(self, fn, *iterables):
        return self.pool.map(_call, [(SEGMENT, fn.__module__, fn.__name__, a) for a in zip(*iterables)])


def run_module(key, seg, out_dir, quick=False):
    global SEGMENT
    name = MODULES[key][0]
    mod = importlib.import_module("experiments." + name)
    SEGMENT = seg
    original = mod.ProcessPoolExecutor
    mod.ProcessPoolExecutor = AuditPool
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        mod.main(out_dir=out_dir, quick=quick)
    finally:
        mod.ProcessPoolExecutor = original
    res = json.loads((out_dir / f"{name}_raw.json").read_text())
    return res, {k: {"ok": bool(ok), "label": lab, "detail": d} for k, lab, ok, d in mod.predictions(res)}


def max_diff(a, b):
    """Largest absolute difference between two JSON trees (inf if their structure differs)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max([max_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max([max_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None:
        return 0.0 if a == b else float("inf")
    if np.isnan(a) and np.isnan(b):
        return 0.0
    return abs(float(a) - float(b))


def e51_real(res):
    """G(turn), G(shape), G(motion) at e51's pre-registered horizon (10 steps) for the six real schools."""
    from experiments.e51_trusted_closure import _sel
    return {t: [r["h"]["10"][t]["conf"]["G"] for r in _sel(res, 8, "real")] for t in ("turn", "shape", "motion")}


def predictions(res):
    pub, rot = res["published"], res["rotations"]
    rots = [rot[s] for s in ("first", "middle")]
    ok = lambda key, p: [r["verdicts"][key][p]["ok"] for r in rots]  # noqa: E731
    out = []
    d = res["reproduction"]
    out.append(("R0", "machinery: the last segment reproduces every published raw result (<= 1e-9)",
                all(v <= 1e-9 for v in d.values()), ", ".join(f"{k} {v:.1e}" for k, v in d.items())))
    moved = [f"{s}:{k}.{p}" for s in ("first", "middle") for k, ps in SIM.items() for p in ps
             if rot[s]["verdicts"][k][p]["ok"] != pub[k][p]["ok"]]
    out.append(("R1", "simulation verdicts unchanged in both rotations", not moved,
                "moved: " + (", ".join(moved) or "none")))
    k3 = ok("e51", "K3")
    out.append(("H1", "the next turn is open: e51 K3 passes in both rotations", all(k3), f"first/middle {k3}"))
    g = [r["e51"] for r in rots]
    sh = [sum(v < 0.05 for v in x["shape"]) for x in g]
    out.append(("H2", "shape is closed: G(shape) < 0.05 in >= 5/6 real schools, both rotations",
                all(c >= 5 for c in sh), f"{sh[0]}/6, {sh[1]}/6; " + " | ".join(
                    " ".join(f"{v:+.3f}" for v in x["shape"]) for x in g)))
    mo = [sum(v < 0.03 for v in x["motion"]) for x in g]
    out.append(("H3", "motion is closed: G(motion) < 0.03 in >= 5/6 real schools, both rotations",
                all(c >= 5 for c in mo), f"{mo[0]}/6, {mo[1]}/6; " + " | ".join(
                    " ".join(f"{v:+.3f}" for v in x["motion"]) for x in g)))
    h4 = ok("e52", "D3") + ok("e53", "Z3")
    out.append(("H4", "the front steers: e52 D3 and e53 Z3 pass in both rotations", all(h4),
                f"D3 {h4[:2]}, Z3 {h4[2:]}"))
    h5 = ok("e52", "D1") + ok("e52", "D4")
    out.append(("H5", "the learned map: e52 D1 and D4 pass in both rotations", all(h5), f"D1 {h5[:2]}, D4 {h5[2:]}"))
    k6 = ok("e51", "K6")
    out.append(("H6", "schools more self-contained than virtual twins: e51 K6 passes in both", all(k6), f"{k6}"))
    p1 = ok("e53", "P1")
    out.append(("H7", "pigeons closed at 1 s: e53 P1 fails in both rotations", not any(p1), f"{p1}"))
    med = [float(np.median(x["turn"])) for x in g]
    out.append(("H8", "size: median e51 G(turn) in [0.06, 0.25] in both rotations",
                all(0.06 <= v <= 0.25 for v in med), "first {:+.3f}, middle {:+.3f}".format(*med)))
    agree = [sum(r["verdicts"][k][p]["ok"] == pub[k][p]["ok"] for k, ps in REAL.items() for p in ps) for r in rots]
    flips = [", ".join(f"{k}.{p}" for k, ps in REAL.items() for p in ps
                       if r["verdicts"][k][p]["ok"] != pub[k][p]["ok"]) or "none" for r in rots]
    out.append(("V1", ">= 20 of 24 real-data verdicts match the published ones, in each rotation",
                all(a >= 20 for a in agree), f"first {agree[0]}/24 (flipped {flips[0]}); "
                f"middle {agree[1]}/24 (flipped {flips[1]})"))
    return out


def analyze(res):
    lines = ["e78: segment rotation audit of the fish closure experiments (e51-e60)", ""]
    for s in ("first", "middle"):
        g = res["rotations"][s]["e51"]
        lines.append(f"{s}: e51 G(turn) " + " ".join(f"{v:+.3f}" for v in g["turn"]))
    lines.append("")
    lines.append("Real-data verdicts (published / first / middle):")
    for k, ps in REAL.items():
        for p in ps:
            v = [res["published"][k][p]["ok"]] + [res["rotations"][s]["verdicts"][k][p]["ok"]
                                                  for s in ("first", "middle")]
            lines.append(f"  {k} {p}: " + " / ".join("PASS" if x else "fail" for x in v))
    lines += ["", "Predictions:"]
    preds = predictions(res)
    for k, lab, ok, d in preds:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {k} {lab}\n         {d}")
    lines.append(f"\n{sum(p[2] for p in preds)}/{len(preds)} predictions pass")
    return "\n".join(lines)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    if quick:
        raise SystemExit("e78 has no quick mode: the reproduction check needs the full runs")
    out_dir = out_dir or Path("results")
    rot_dir = out_dir / "e78_rotations"
    published, reproduction, rotations = {}, {}, {}
    for key, (name, _) in MODULES.items():
        res = json.loads((out_dir / f"{name}_raw.json").read_text())
        mod = importlib.import_module("experiments." + name)
        published[key] = {k: {"ok": bool(ok), "label": lab, "detail": d} for k, lab, ok, d in mod.predictions(res)}
        new, _ = run_module(key, SEGMENTS["last"], rot_dir / "last" / key)
        reproduction[key] = max_diff(res["runs"], new["runs"])
        if key == "e51":
            published_e51 = e51_real(res)
    for s in ("first", "middle"):
        rotations[s] = {"verdicts": {}}
        for key in MODULES:
            new, verdicts = run_module(key, SEGMENTS[s], rot_dir / s / key)
            rotations[s]["verdicts"][key] = verdicts
            if key == "e51":
                rotations[s]["e51"] = e51_real(new)
    res = {"published": published, "published_e51": published_e51, "reproduction": reproduction,
           "rotations": rotations, "segments": SEGMENTS}
    (out_dir / "e78_split_audit_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    (out_dir / "e78_split_audit.txt").write_text(text)
    from experiments.e78_figures import make_figures
    make_figures(res, out_dir / "e78_figures", text)
    return text


if __name__ == "__main__":
    print(main())
