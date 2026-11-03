"""Experiment 54: are levels closed in their stable variables and open along their neutral directions?

e51-e52 found real fish schools closed in their shape and motion, but open in
their next turn, which a few fast front fish carry for about a second. The
white paper (Section 7.6) proposes why: shape and spread are held in place
by interaction (disturb them and they return), while the school's heading
is a neutral direction (every heading is equally good), so whatever is
happening inside the school can steer it.

e54 tests that explanation by intervention, in two simulated systems where
everything is known:
  flock     30 agents in open space: k = 6 nearest alignment and attraction,
            weak cohesion, repulsion, heterogeneous cruising speeds with
            speed matching and front-slowing (e54_sims.flock).
  kuramoto  30 noisy oscillators, heterogeneous natural frequencies,
            K = 1.5 (r about 0.9) (e54_sims.kuramoto).
An external field f sin(psi0 - theta_i), psi0 = 0, is added to every member,
f in {0, 0.05, 0.1, 0.2, 0.4}: it turns the group's direction from a neutral
into a stable variable. Everything else is unchanged. Two seeds per
setting; 25 000 steps of 0.1 after a burn-in; first 70% fitted, last 30%
measured. Only cohesion, polarization, r and the field's pinning of psi
were checked before the predictions were written (no closure numbers).

Test: e51's calibrated closure gap G at h = 10 steps (one time unit) with
e51's cross-validated ridge and block bootstrap.
  block    the group state and its last 5 values, quadratic terms, and the
           group's last 5 turning rates. Flock: mean heading vector, mean
           speed, polarization, spread, milling (no centroid: open space).
           Kuramoto: r, cos psi, sin psi.
  extra    the arrangement inside the group. Flock: e52's 75 mirror-symmetric
           influence features. Kuramoto: means over oscillators of sin k phi,
           cos k phi (k = 1..3), u, u cos k phi, u sin k phi (k = 1, 2), u^2
           and u^2 sin phi, with phi = theta - psi and u the oscillator's last
           phase step relative to psi's (tamed), and their squares.
  targets  turn: the change of psi; stable: the change of (polarization,
           spread), whitened (flock), or of r (Kuramoto).
  checks   null: the arrangement of another seed's run in the same setting;
           planted: e52's 5% signal in the turn target.

Predictions, fixed before running (per system, 10 runs = 5 fields x 2 seeds):
  N0  The test works: planted rise >= 0.025 with lower bound > 0 in >= 8/10,
      and null |G| <= 0.01 for both targets in >= 8/10.
  N1  The neutral direction is open: with no field, G(turn) >= 0.03 with
      lower bound > 0, in both seeds.
  N2  The stable variables are closed: G(stable) <= 0.02 in >= 9/10 runs.
  N3  Pinning closes the direction: mean G(turn) at f = 0.4 <= one third of
      mean G(turn) at f = 0.
  N4  The stronger the pinning, the more closed: Spearman correlation of f
      with G(turn) over the 10 runs <= -0.7.
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
from experiments.e52_influence_map import _tame_fish, bumps, influence
from experiments.e54_sims import flock, kuramoto

SEED = 54
FIELDS = (0.0, 0.05, 0.1, 0.2, 0.4)
SEEDS = (0, 1)
H = 10
STEPS = 25000


def _lags(x):
    return np.concatenate([np.roll(x, j, axis=0) for j in range(LAGS)], 1)


def _quad(x):
    iu = np.triu_indices(x.shape[1])
    return (x[:, :, None] * x[:, None, :])[:, iu[0], iu[1]]


def _rates(psi):
    return np.column_stack([np.roll(_wrap(psi - np.roll(psi, 1)), j) for j in range(5)])


def flock_parts(seed, field, ref):
    p, th, sp = flock(steps=STEPS, field=field, seed=seed)
    z = school_state(p, th, sp)[:, 2:]
    psi = np.arctan2(z[:, 1], z[:, 0])
    block = np.concatenate([_lags(z), _quad(z), _rates(psi)], 1)
    extra, _, _ = influence(p, th, sp, ref)
    stable = z[:, [3, 4]]
    return block, extra, psi, stable


def kuramoto_arrangement(th, ref):
    zc = np.exp(1j * th).mean(1)
    psi = np.angle(zc)
    phi = _wrap(th - psi[:, None])
    u = _tame_fish(_wrap(_wrap(th - np.roll(th, 1, axis=0)) - _wrap(psi - np.roll(psi, 1))[:, None]), ref)
    m = lambda a: a.mean(1)  # noqa: E731
    cols = [m(f(k * phi)) for k in (1, 2, 3) for f in (np.sin, np.cos)]
    cols += [m(u)] + [m(u * f(k * phi)) for k in (1, 2) for f in (np.cos, np.sin)] + [m(u * u), m(u * u * np.sin(phi))]
    a = np.column_stack(cols)
    return np.concatenate([a, a ** 2], 1), zc


def kuramoto_parts(seed, field, ref):
    th = kuramoto(steps=STEPS, field=field, seed=seed)
    extra, zc = kuramoto_arrangement(th, ref)
    psi = np.angle(zc)
    z = np.column_stack([np.abs(zc), np.cos(psi), np.sin(psi)])
    block = np.concatenate([_lags(z), _quad(z), _rates(psi)], 1)
    return block, extra, psi, z[:, [0]]


def _run(args):
    system, field, seed = args
    rng = np.random.default_rng(SEED * 1000 + 100 * FIELDS.index(field) + 10 * seed + (system == "kuramoto"))
    parts = flock_parts if system == "flock" else kuramoto_parts
    cut = int(0.7 * STEPS)
    ref = np.arange(cut)
    block, extra, psi, stable = parts(seed, field, ref)
    null = parts(seed + 100, field, ref)[1]
    rows = np.arange(LAGS + 1, STEPS - H)
    train, test = rows[rows + H < cut], rows[rows >= cut]
    rows = np.concatenate([train, test])
    pos_tr, pos_te = np.arange(len(train)), np.arange(len(train), len(rows))
    nb = block.shape[1]
    targets = {"turn": _wrap(psi[rows + H] - psi[rows])[:, None], "stable": (stable[rows + H] - stable[rows])}
    models = {"block": Ridge(block, train, test, H + LAGS),
              "arrangement": Ridge(np.concatenate([block, extra], 1), train, test, H + LAGS, nb),
              "null": Ridge(np.concatenate([block, null], 1), train, test, H + LAGS, nb)}
    out = {"system": system, "field": field, "seed": seed,
           "psi_sd": float(np.std(np.unwrap(psi))), "turn_sd": float(targets["turn"].std())}
    for name, y in targets.items():
        y_tr, y_te = y[pos_tr], y[pos_te]
        w = 1 / y_tr.var(0)
        vinv = np.linalg.inv(np.atleast_2d(np.cov(y_te.T)))
        q = {}
        for k, m in models.items():
            pr, lam, s = m.fit(y_tr, w)
            e = y_te - pr
            q[k] = np.einsum("ti,ij,tj->t", e, vinv, e)
            out.setdefault("scale", {})[f"{name}_{k}"] = s
        out[name] = {k: gap_ci(q[k], q["block"], rng) for k in ("arrangement", "null")}
    ones_block = np.column_stack([np.ones(len(psi)), block])
    j0 = bumps().index(("sin", -1, 0.75, 0.0)) if system == "flock" else 8
    xi = extra[:, j0] - ones_block @ np.linalg.lstsq(ones_block[rows], extra[rows, j0], rcond=None)[0]
    y = targets["turn"][:, 0]
    one = np.ones(1)
    pb = models["block"].fit(y[pos_tr, None], one)[0][:, 0]
    kappa = np.sqrt(PLANT / (1 - PLANT) * ((y[pos_te] - pb) ** 2).mean() / xi[test].var())
    yp = y + kappa * xi[rows]
    fb = models["block"].fit(yp[pos_tr, None], one)[0][:, 0]
    fa = models["arrangement"].fit(yp[pos_tr, None], one)[0][:, 0]
    out["planted"] = {"before": out["turn"]["arrangement"],
                      "after": gap_ci((yp[pos_te] - fa) ** 2, (yp[pos_te] - fb) ** 2, rng)}
    return out


def _sel(res, system):
    return sorted([r for r in res["runs"] if r["system"] == system], key=lambda r: (r["field"], r["seed"]))


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def predictions(res):
    out = []
    for system, key in (("flock", "F"), ("kuramoto", "K")):
        rs = _sel(res, system)
        cal = sum(r["planted"]["after"]["G"] - r["planted"]["before"]["G"] >= 0.025 and r["planted"]["after"]["lo"] > 0
                  for r in rs)
        nul = sum(abs(r["turn"]["null"]["G"]) <= 0.01 and abs(r["stable"]["null"]["G"]) <= 0.01 for r in rs)
        g0 = [r["turn"]["arrangement"] for r in rs if r["field"] == 0.0]
        n1 = sum(g["G"] >= 0.03 and g["lo"] > 0 for g in g0)
        n2 = sum(r["stable"]["arrangement"]["G"] <= 0.02 for r in rs)
        m0 = np.mean([g["G"] for g in g0])
        m4 = np.mean([r["turn"]["arrangement"]["G"] for r in rs if r["field"] == max(FIELDS)])
        rho = _spearman([r["field"] for r in rs], [r["turn"]["arrangement"]["G"] for r in rs])
        turn = ", ".join(f"{r['field']:g}:{r['turn']['arrangement']['G']:+.3f}" for r in rs)
        out += [
            (f"{key}0", f"{system}: the test works (planted >= 8/10, null >= 8/10)", cal >= 8 and nul >= 8,
             f"planted {cal}/10, null {nul}/10"),
            (f"{key}1", f"{system}: neutral direction open with no field, G(turn) >= 0.03, lower > 0 (2/2)", n1 == 2,
             f"{n1}/2; " + ", ".join(f"{g['G']:+.3f} [{g['lo']:+.3f}]" for g in g0)),
            (f"{key}2", f"{system}: stable variables closed, G(stable) <= 0.02 (>= 9/10)", n2 >= 9,
             f"{n2}/10; " + ", ".join(f"{r['stable']['arrangement']['G']:+.3f}" for r in rs)),
            (f"{key}3", f"{system}: pinning closes the direction, G(turn, f=0.4) <= G(turn, f=0) / 3", m4 <= m0 / 3,
             f"{m0:+.3f} -> {m4:+.3f}"),
            (f"{key}4", f"{system}: stronger pinning, more closed: Spearman(f, G(turn)) <= -0.7", rho <= -0.7,
             f"rho {rho:+.2f}; {turn}"),
        ]
    return out


def analyze(res) -> str:
    lines = ["Experiment 54: stable variables close, neutral directions stay open? (pinning field test)", ""]
    for r in sorted(res["runs"], key=lambda r: (r["system"], r["field"], r["seed"])):
        c = lambda k, t: f"{r[t][k]['G']:+.3f} [{r[t][k]['lo']:+.3f},{r[t][k]['hi']:+.3f}]"  # noqa: E731
        lines.append(f"  {r['system']:8} f={r['field']:<4g} seed {r['seed']}  psi sd {r['psi_sd']:6.2f}  "
                     f"turn sd {r['turn_sd']:.3f} | G turn {c('arrangement', 'turn')}  G stable {c('arrangement', 'stable')}"
                     f" | null {r['turn']['null']['G']:+.3f}/{r['stable']['null']['G']:+.3f}"
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
    jobs = [(s, f, k) for s in ("flock", "kuramoto") for f in FIELDS for k in SEEDS]
    if quick:
        jobs = [(s, f, 0) for s in ("flock", "kuramoto") for f in (0.0, 0.4)]
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run, jobs))
    res = {"runs": runs, "fields": list(FIELDS)}
    (out_dir / "e54_neutral_directions_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e54_figures import make_figures
    make_figures(res, out_dir / "e54_figures", text)
    return text


if __name__ == "__main__":
    main()
