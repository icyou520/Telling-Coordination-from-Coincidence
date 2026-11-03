"""Experiment 55: is a level open exactly where some of its members are not locked to it?

e54 rejected "stable variables close, neutral directions stay open". After the fact, its open cases shared
something else: a few members not slaved to the collective. In the oscillators, how open r was tracked the
number of phase slips of one or two unlocked oscillators almost perfectly. e55 tests that idea directly, with
no field, by setting the number m of free members in {0, 1, 2, 4, 8} out of 30 (e55_sims):
  kuramoto  free members have natural frequencies far outside the locking range and drift through the group;
            the others are tightly locked (about one slip per run without free members).
  flock     e54's flock; free members also steer towards private goal directions that wander slowly
            (informed individuals whose information no one else has).
Two seeds per m; 25 000 steps of 0.1; first 70% fitted, last 30% measured. Only r, slips, polarization,
spread and turn sizes were checked before the predictions were written (no closure numbers).

Test: e54's, unchanged (calibrated closure gap G at one time unit, e51's ridge and bootstrap), with the
same block and arrangement features. The affected variable is r for the oscillators (as found in e54) and
the turn for the flock (as in e52 and e54); the other variable is reported too. Two extra models compute the
arrangement features over the free members only, or over the locked members only (m >= 1).

Predictions, fixed before running (per system, 10 runs = 5 values of m x 2 seeds), G for the affected
variable:
  L0  The test works: planted rise >= 0.025 with lower bound > 0 in >= 8/10, null |G| <= 0.01 (both
      variables) in >= 8/10.
  L1  More free members, more open: Spearman(m, G) >= 0.7.
  L2  The openness is the free members': mean G at m = 8 >= mean G at m = 0 + 0.03.
  L3  The information is in the free members: G(free members' features) >= G(locked members' features) in
      >= 7/8 runs with m >= 1.
  K4  Oscillators, nothing free, nothing open: G(r) <= 0.02 in both runs with m = 0.
  F4  Flock, the shape stays closed: G(polarization, spread) <= 0.02 in >= 9/10.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e49_school_layer import LAGS, school_state
from experiments.e50_school_configuration import _wrap
from experiments.e51_trusted_closure import PLANT, Ridge, gap_ci
from experiments.e52_influence_map import _bump, _tame_fish, bumps, fish_frame
from experiments.e54_neutral_directions import _lags, _quad, _rates, _spearman
from experiments.e55_sims import flock_free, kuramoto_free

SEED = 55
MS = (0, 1, 2, 4, 8)
SEEDS = (0, 1)
H = 10
STEPS = 25000
N = 30
AFFECTED = {"kuramoto": "stable", "flock": "turn"}


def influence_sets(p, th, sp, ref, sets):
    """e52's 75 influence features, averaged over each index set in sets."""
    x, y, phi, omega, u = fish_frame(p, th, sp)
    om, ut = _tame_fish(omega, ref), _tame_fish(u, ref)
    ch = {"place": np.ones_like(x), "sin": np.sin(phi), "cos": np.cos(phi), "turn": om, "speed": ut,
          "speed_sin": ut * np.sin(phi)}
    per = [ch[c] * (_bump(x, y, a, b) - par * _bump(x, -y, a, b)) for c, par, a, b in bumps()]
    return [np.column_stack([f[:, idx].mean(1) for f in per]) for idx in sets]


def phase_sets(th, ref, sets):
    """e54's oscillator arrangement features, averaged over each index set in sets."""
    psi = np.angle(np.exp(1j * th).mean(1))
    phi = _wrap(th - psi[:, None])
    u = _tame_fish(_wrap(_wrap(th - np.roll(th, 1, axis=0)) - _wrap(psi - np.roll(psi, 1))[:, None]), ref)
    out = []
    for idx in sets:
        f, w = phi[:, idx], u[:, idx]
        m = lambda a: a.mean(1)  # noqa: E731
        cols = [m(g(k * f)) for k in (1, 2, 3) for g in (np.sin, np.cos)]
        cols += [m(w)] + [m(w * g(k * f)) for k in (1, 2) for g in (np.cos, np.sin)] + [m(w * w), m(w * w * np.sin(f))]
        a = np.column_stack(cols)
        out.append(np.concatenate([a, a ** 2], 1))
    return out


def parts(system, m, seed, ref):
    sets = [np.arange(N), np.arange(m), np.arange(m, N)] if m else [np.arange(N)]
    if system == "flock":
        p, th, sp = flock_free(m=m, steps=STEPS, seed=seed)
        z = school_state(p, th, sp)[:, 2:]
        psi = np.arctan2(z[:, 1], z[:, 0])
        stable = z[:, [3, 4]]
        feats = influence_sets(p, th, sp, ref, sets)
    else:
        th = kuramoto_free(m=m, steps=STEPS, seed=seed)
        zc = np.exp(1j * th).mean(1)
        psi = np.angle(zc)
        z = np.column_stack([np.abs(zc), np.cos(psi), np.sin(psi)])
        stable = z[:, [0]]
        feats = phase_sets(th, ref, sets)
    block = np.concatenate([_lags(z), _quad(z), _rates(psi)], 1)
    return block, feats, psi, stable


def _run(args):
    system, m, seed = args
    rng = np.random.default_rng(SEED * 1000 + 100 * MS.index(m) + 10 * seed + (system == "kuramoto"))
    cut = int(0.7 * STEPS)
    ref = np.arange(cut)
    block, feats, psi, stable = parts(system, m, seed, ref)
    null = parts(system, m, seed + 100, ref)[1][0]
    rows = np.arange(LAGS + 1, STEPS - H)
    train, test = rows[rows + H < cut], rows[rows >= cut]
    rows = np.concatenate([train, test])
    pos_tr, pos_te = np.arange(len(train)), np.arange(len(train), len(rows))
    nb = block.shape[1]
    targets = {"turn": _wrap(psi[rows + H] - psi[rows])[:, None], "stable": (stable[rows + H] - stable[rows])}
    extras = {"arrangement": feats[0], "null": null}
    if m:
        extras.update(free=feats[1], locked=feats[2])
    models = {"block": Ridge(block, train, test, H + LAGS)}
    models.update({k: Ridge(np.concatenate([block, x], 1), train, test, H + LAGS, nb) for k, x in extras.items()})
    out = {"system": system, "m": m, "seed": seed, "turn_sd": float(targets["turn"].std())}
    for name, y in targets.items():
        y_tr, y_te = y[pos_tr], y[pos_te]
        w = 1 / y_tr.var(0)
        vinv = np.linalg.inv(np.atleast_2d(np.cov(y_te.T)))
        q = {}
        for k, mod in models.items():
            e = y_te - mod.fit(y_tr, w)[0]
            q[k] = np.einsum("ti,ij,tj->t", e, vinv, e)
        out[name] = {k: gap_ci(q[k], q["block"], rng) for k in extras}
    aff = AFFECTED[system]
    y = targets[aff][:, 0]
    extra = feats[0]
    ones_block = np.column_stack([np.ones(len(psi)), block])
    j0 = bumps().index(("sin", -1, 0.75, 0.0)) if system == "flock" else 2
    xi = extra[:, j0] - ones_block @ np.linalg.lstsq(ones_block[rows], extra[rows, j0], rcond=None)[0]
    one = np.ones(1)
    pb = models["block"].fit(y[pos_tr, None], one)[0][:, 0]
    kappa = np.sqrt(PLANT / (1 - PLANT) * ((y[pos_te] - pb) ** 2).mean() / xi[test].var())
    yp = y + kappa * xi[rows]
    fb = models["block"].fit(yp[pos_tr, None], one)[0][:, 0]
    fa = models["arrangement"].fit(yp[pos_tr, None], one)[0][:, 0]
    before = gap_ci((y[pos_te] - models["arrangement"].fit(y[pos_tr, None], one)[0][:, 0]) ** 2,
                    (y[pos_te] - pb) ** 2, rng)
    out["planted"] = {"before": before, "after": gap_ci((yp[pos_te] - fa) ** 2, (yp[pos_te] - fb) ** 2, rng)}
    return out


def _sel(res, system):
    return sorted([r for r in res["runs"] if r["system"] == system], key=lambda r: (r["m"], r["seed"]))


def predictions(res):
    out = []
    for system, key in (("flock", "F"), ("kuramoto", "K")):
        rs = _sel(res, system)
        aff = AFFECTED[system]
        g = [r[aff]["arrangement"]["G"] for r in rs]
        cal = sum(r["planted"]["after"]["G"] - r["planted"]["before"]["G"] >= 0.025 and r["planted"]["after"]["lo"] > 0
                  for r in rs)
        nul = sum(abs(r["turn"]["null"]["G"]) <= 0.01 and abs(r["stable"]["null"]["G"]) <= 0.01 for r in rs)
        rho = _spearman([r["m"] for r in rs], g)
        m0 = np.mean([v for r, v in zip(rs, g) if r["m"] == 0])
        m8 = np.mean([v for r, v in zip(rs, g) if r["m"] == max(MS)])
        fr = [r for r in rs if r["m"]]
        l3 = sum(r[aff]["free"]["G"] >= r[aff]["locked"]["G"] for r in fr)
        name = "r" if system == "kuramoto" else "turn"
        out += [
            (f"{key}0", f"{system}: the test works (planted >= 8/10, null >= 8/10)", cal >= 8 and nul >= 8,
             f"planted {cal}/10, null {nul}/10"),
            (f"{key}1", f"{system}: more free members, more open: Spearman(m, G({name})) >= 0.7", rho >= 0.7,
             f"rho {rho:+.2f}; " + ", ".join(f"{r['m']}:{v:+.3f}" for r, v in zip(rs, g))),
            (f"{key}2", f"{system}: mean G({name}) at m=8 >= mean at m=0 + 0.03", m8 >= m0 + 0.03,
             f"{m0:+.3f} -> {m8:+.3f}"),
            (f"{key}3", f"{system}: information in the free members: G(free) >= G(locked) (>= 7/8)", l3 >= 7,
             f"{l3}/8; " + ", ".join(f"{r['m']}:{r[aff]['free']['G']:+.3f}/{r[aff]['locked']['G']:+.3f}" for r in fr)),
        ]
        if system == "kuramoto":
            z = [r["stable"]["arrangement"]["G"] for r in rs if r["m"] == 0]
            out.append(("K4", "kuramoto: nothing free, nothing open: G(r) <= 0.02 at m=0 (2/2)",
                        all(v <= 0.02 for v in z), ", ".join(f"{v:+.3f}" for v in z)))
        else:
            s = [r["stable"]["arrangement"]["G"] for r in rs]
            out.append(("F4", "flock: the shape stays closed: G(polarization, spread) <= 0.02 (>= 9/10)",
                        sum(v <= 0.02 for v in s) >= 9, ", ".join(f"{v:+.3f}" for v in s)))
    return out


def analyze(res) -> str:
    lines = ["Experiment 55: is a level open exactly where some members are not locked to it?", ""]
    for r in sorted(res["runs"], key=lambda r: (r["system"], r["m"], r["seed"])):
        c = lambda t, k: f"{r[t][k]['G']:+.3f} [{r[t][k]['lo']:+.3f},{r[t][k]['hi']:+.3f}]"  # noqa: E731
        fl = "".join(f" | {t} free/locked {r[t]['free']['G']:+.3f}/{r[t]['locked']['G']:+.3f}"
                     for t in ("turn", "stable")) if r["m"] else ""
        lines.append(f"  {r['system']:8} m={r['m']} seed {r['seed']}  G turn {c('turn', 'arrangement')}  "
                     f"G stable {c('stable', 'arrangement')}{fl} | null {r['turn']['null']['G']:+.3f}/"
                     f"{r['stable']['null']['G']:+.3f} | planted {r['planted']['before']['G']:+.3f} -> "
                     f"{r['planted']['after']['G']:+.3f}")
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
    jobs = [(s, m, k) for s in ("flock", "kuramoto") for m in MS for k in SEEDS]
    if quick:
        jobs = [(s, m, 0) for s in ("flock", "kuramoto") for m in (0, 8)]
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run, jobs))
    res = {"runs": runs, "ms": list(MS)}
    (out_dir / "e55_free_members_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e55_figures import make_figures
    make_figures(res, out_dir / "e55_figures", text)
    return text


if __name__ == "__main__":
    main()
