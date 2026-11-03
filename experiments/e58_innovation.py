"""Experiment 58: is a flock's or school's openness carried by momentary innovations of its members?

e55-e57: in oscillators, openness sits in members that are measurably loose (a fixed trait). In flocks and
fish no member is loose in heading; privately informed agents changed little, and their influence reached
their neighbours within the forecast horizon. e58 tests the idea that in such groups freedom is momentary:
a member briefly turns in a way its neighbourhood does not explain, and that innovation spreads.

Innovation: for each member and step, the residual of its heading change after a local model fitted
(pooled over members, ordinary least squares, fitted part only) on what its surroundings explain:
alignment, attraction and repulsion with its nearest neighbours (e31's social terms), the direction to the
group's centre, walls (real fish), its own previous turn and its relative speed. A member's independence
iota_i = var(innovation) / var(heading change) over the fitted part. The innovation map: the last 5 steps'
mean innovation of each member, tamed, placed with e52's bumps in the school's frame (odd channel, 15
features, mirror-symmetric).

Data: 24 new flocks (e55_sims.flock_free, 30 agents) drawn at random with 1-4 privately informed agents,
noise in [0.2, 0.6], alignment in [0.8, 2.0], speed spread in [0.05, 0.3]; the 6 Puy et al. 8-fish
schools and the 9 zebrafish groups. Nothing about innovations or independence was looked at before the
predictions were written.

Test: e56's (closure gap G for the group's turn at one time unit / 1 s; e51's ridge; bootstrap) with extras:
  innov    the innovation map (all members)
  all      e52's 75 influence features (all members)
  ownturn  e52's own-turning channel only (raw turning, not innovation)
  indep / conform   e52's features over the k most / least independent members (k = 3; fish max(2, N/4))
  null     the innovation map of another flock (same parameters, other seed) or fish group
Planted (flocks): 5% in the turn, built from an innovation-map feature.

Predictions, fixed before running:
  V1  The measure finds the informed: informed agents are more independent than the others
      (AUC >= 0.8) in >= 20/24 flocks.
  S0  The test works in the flocks: planted rise >= 0.025 with lower bound > 0 in >= 19/24, null
      |G| <= 0.01 in >= 19/24.
  I1  Flocks: innovations carry the openness: where the turn is open (G_all >= 0.03, lower > 0),
      G_innov >= 0.5 G_all in >= 80%.
  I2  Flocks: innovations beat raw turning: G_innov > G_ownturn in >= 80% of open flocks.
  I3  Flocks: the independent members carry more: G_indep > G_conform in >= 80% of open flocks.
  F1  Fish: G_innov >= 0.5 G_all in >= 80% of open real groups.
  F2  Fish: G_innov > G_ownturn in >= 11/15.
  F3  Fish: G_indep > G_conform in >= 11/15.
  F4  Fish: null clean (|G| <= 0.01) in >= 12/15.
"""

from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import social_terms
from experiments.e49_school_layer import LAGS, school_state
from experiments.e50_school_configuration import _wrap
from experiments.e52_influence_map import BX, BY, _bump, _tame_fish, bumps, fish_frame
from experiments.e53_influence_replication import ZEBRA, _rows, gap_ci, place_map
from experiments.e52_influence_map import null_recordings
from experiments.e54_neutral_directions import _lags, _quad, _rates
from experiments.e55_sims import flock_free
from experiments.e56_measured_lock import _closure, influence_sets

SEED = 58
RUNS = 24
STEPS = 25000
H = 10
K_SIM = 3
SMOOTH = 5
OWN = [j for j, (c, *_) in enumerate(bumps()) if c == "turn"]


def draw(i):
    rng = np.random.default_rng(SEED * 100 + i)
    return {"noise": float(rng.uniform(0.2, 0.6)), "align": float(rng.uniform(0.8, 2.0)),
            "speed_sd": float(rng.uniform(0.05, 0.3)), "m": int(rng.integers(1, 5))}


def innovation(p, th, sp, walls, ref):
    """Innovations (T, N), tamed and smoothed over the last SMOOTH steps, and each member's independence."""
    d = np.zeros_like(th)
    d[1:] = _wrap(th[1:] - th[:-1])
    g = p.mean(1, keepdims=True) - p
    centre = np.sin(np.arctan2(g[..., 1], g[..., 0]) - th)
    u = sp / np.maximum(sp.mean(1, keepdims=True), 1e-6) - 1
    parts = [np.ones(th.shape + (1,)), social_terms(p, th), centre[..., None], d[..., None], u[..., None]]
    if walls is not None:
        parts.append(walls)
    x = np.concatenate(parts, -1)
    xs = np.roll(x, 1, axis=0)
    fit = ref[2:]
    coef = np.linalg.lstsq(xs[fit].reshape(-1, x.shape[-1]), d[fit].ravel(), rcond=None)[0]
    eps = d - xs @ coef
    eps[:2] = 0.0
    iota = eps[fit].var(0) / np.maximum(d[fit].var(0), 1e-12)
    e = _tame_fish(eps, fit)
    sm = np.cumsum(e, 0)
    sm[SMOOTH:] = sm[SMOOTH:] - sm[:-SMOOTH]
    return sm / SMOOTH, iota


def innov_map(p, th, sp, e):
    x, y, *_ = fish_frame(p, th, sp)
    return np.column_stack([(e * (_bump(x, y, a, b) + _bump(x, -y, a, b))).mean(1) for a in BX for b in BY])


def _sets(iota, k):
    order = np.argsort(iota)
    return [np.arange(len(iota)), order[-k:], order[:k]]


def _flock(i, seed, ref):
    prm = draw(i)
    p, th, sp = flock_free(steps=STEPS, seed=seed, **prm)
    e, iota = innovation(p, th, sp, None, ref)
    return prm, p, th, sp, e, iota


def _auc(score, pos):
    a, b = score[pos], score[~pos]
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def _run_flock(i):
    rng = np.random.default_rng(SEED * 1000 + i)
    cut = int(0.7 * STEPS)
    ref = np.arange(cut)
    prm, p, th, sp, e, iota = _flock(i, 0, ref)
    n = th.shape[1]
    feats = influence_sets(p, th, sp, ref, _sets(iota, K_SIM))
    innov = innov_map(p, th, sp, e)
    _, po, tho, spo, eo, _ = _flock(i, 100, ref)
    null = innov_map(po, tho, spo, eo)
    z = school_state(p, th, sp)[:, 2:]
    psi = np.arctan2(z[:, 1], z[:, 0])
    block = np.concatenate([_lags(z), _quad(z), _rates(psi)], 1)
    rows = np.arange(LAGS + SMOOTH + 2, STEPS - H)
    train, test = rows[rows + H < cut], rows[rows >= cut]
    rows = np.concatenate([train, test])
    targets = {"turn": _wrap(psi[rows + H] - psi[rows])[:, None]}
    extras = {"all": innov, "every": feats[0], "ownturn": feats[0][:, OWN], "indep": feats[1],
              "conform": feats[2], "null": null}
    j0 = BY.index(0.0) + len(BY) * BX.index(0.75)
    out = _closure(block, extras, targets, train, test, H + LAGS, rng, "turn", innov[:, j0])
    names = {"all": "innov", "every": "all"}
    out["turn"] = {names.get(k, k): v for k, v in out["turn"].items()}
    informed = np.arange(n) < prm["m"]
    return {"system": "flock", "i": i, "params": prm, "iota": iota.tolist(), "auc": _auc(iota, informed), **out}


def _fish(spec):
    if spec[0] == "puy":
        from experiments.e31_real_fish import kinematics, wall_terms
        from experiments.fish_data import load
        p, th, sp = kinematics(load(8, spec[1]))
        sp = np.where(sp > 100.0, 0.0, sp)
        brk = np.zeros(len(th), bool)
        brk[0] = True
        return p, th, sp, brk, 0.1, wall_terms(p, th)
    from experiments.e38_zebrafish import load, wall_terms
    from experiments.e53_influence_replication import zebrafish
    p, th, sp, brk, dt = zebrafish(spec[1], spec[2], "real")
    radius = load(spec[1], spec[2])[1]
    return p, th, sp, brk, dt, wall_terms(p, th, radius)


def _run_fish(args):
    spec, null_spec = args
    rng = np.random.default_rng(SEED * 1000 + zlib.crc32(repr(spec).encode()) % 997)
    p, th, sp, brk, dt, walls = _fish(spec)
    T, n = th.shape
    ref = np.arange(int(0.7 * T))
    e, iota = innovation(p, th, sp, walls, ref)
    k = max(2, n // 4)
    feats = influence_sets(p, th, sp, ref, _sets(iota, k))
    innov = innov_map(p, th, sp, e)
    po, tho, spo, _, _, wo = _fish(null_spec)
    eo, _ = innovation(po, tho, spo, wo, np.arange(int(0.7 * len(tho))))
    null = np.resize(innov_map(po, tho, spo, eo), innov.shape)
    z = school_state(p, th, sp)
    psi = np.arctan2(z[:, 3], z[:, 2])
    block = np.concatenate([_lags(z), _quad(z), place_map(z, ref), _rates(psi)], 1)
    h = int(round(1 / dt))
    brk2 = brk.copy()
    brk2[:SMOOTH + 2] = True
    train, test = _rows(brk2, T, h)
    targets = {"turn": _wrap(psi[np.concatenate([train, test]) + h] - psi[np.concatenate([train, test])])[:, None]}
    extras = {"innov": innov, "all": feats[0], "ownturn": feats[0][:, OWN], "indep": feats[1],
              "conform": feats[2], "null": null}
    out = _closure(block, extras, targets, train, test, h + LAGS, rng, "turn", None)
    return {"system": "fish", "group": f"{spec[0]} {'-'.join(map(str, spec[1:]))}", "N": n, "k": k,
            "iota": iota.tolist(), **out}


def _sel(res, system):
    return sorted([r for r in res["runs"] if r["system"] == system], key=lambda r: r.get("i", 0))


def _open(r):
    return r["turn"]["all"]["G"] >= 0.03 and r["turn"]["all"]["lo"] > 0


def predictions(res):
    fl, fish = _sel(res, "flock"), _sel(res, "fish")
    g = lambda r, k: r["turn"][k]["G"]  # noqa: E731
    v1 = sum(r["auc"] >= 0.8 for r in fl)
    cal = sum(r["planted"]["after"]["G"] - r["planted"]["before"]["G"] >= 0.025 and r["planted"]["after"]["lo"] > 0
              for r in fl)
    nul = sum(abs(g(r, "null")) <= 0.01 for r in fl)
    op = [r for r in fl if _open(r)]
    i1 = sum(g(r, "innov") >= 0.5 * g(r, "all") for r in op)
    i2 = sum(g(r, "innov") > g(r, "ownturn") for r in op)
    i3 = sum(g(r, "indep") > g(r, "conform") for r in op)
    fo = [r for r in fish if _open(r)]
    f1 = sum(g(r, "innov") >= 0.5 * g(r, "all") for r in fo)
    f2 = sum(g(r, "innov") > g(r, "ownturn") for r in fish)
    f3 = sum(g(r, "indep") > g(r, "conform") for r in fish)
    f4 = sum(abs(g(r, "null")) <= 0.01 for r in fish)
    nf = len(fish)
    return [
        ("V1", "the measure finds the informed: AUC >= 0.8 in >= 20/24 flocks", v1 >= 20,
         f"{v1}/24; AUC " + ", ".join(f"{r['auc']:.2f}" for r in fl)),
        ("S0", "the test works in the flocks (planted >= 19/24, null >= 19/24)", cal >= 19 and nul >= 19,
         f"planted {cal}/24, null {nul}/24"),
        ("I1", "flocks: G_innov >= 0.5 G_all where open (>= 80%)", bool(op) and i1 >= 0.8 * len(op), f"{i1}/{len(op)}"),
        ("I2", "flocks: G_innov > G_ownturn where open (>= 80%)", bool(op) and i2 >= 0.8 * len(op), f"{i2}/{len(op)}"),
        ("I3", "flocks: G_indep > G_conform where open (>= 80%)", bool(op) and i3 >= 0.8 * len(op), f"{i3}/{len(op)}"),
        ("F1", "fish: G_innov >= 0.5 G_all where open (>= 80%)", bool(fo) and f1 >= 0.8 * len(fo), f"{f1}/{len(fo)}"),
        ("F2", f"fish: G_innov > G_ownturn in >= 11/{nf}", f2 >= 11, f"{f2}/{nf}"),
        ("F3", f"fish: G_indep > G_conform in >= 11/{nf}", f3 >= 11, f"{f3}/{nf}"),
        ("F4", f"fish: null clean in >= 12/{nf}", f4 >= 12, f"{f4}/{nf}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 58: is openness carried by members' momentary innovations?", ""]
    for r in _sel(res, "flock") + _sel(res, "fish"):
        name = r.get("group") or f"flock {r['i']:2d} m={r['params']['m']}"
        t = r["turn"]
        auc = f"AUC {r['auc']:.2f}" if "auc" in r else " " * 8
        pl = f" | planted {r['planted']['before']['G']:+.3f} -> {r['planted']['after']['G']:+.3f}" if "planted" in r else ""
        lines.append(f"  {name:16} {auc} | all {t['all']['G']:+.3f} [{t['all']['lo']:+.3f}]  innov {t['innov']['G']:+.3f}"
                     f"  ownturn {t['ownturn']['G']:+.3f}  indep {t['indep']['G']:+.3f}  conform {t['conform']['G']:+.3f}"
                     f"  null {t['null']['G']:+.3f}{pl}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def tasks():
    nulls = null_recordings(8, 6)
    fish = [(("puy", r), ("puy", nulls[r])) for r in range(6)]
    fish += [(("zebrafish", n, r), ("zebrafish", *ZEBRA[(j + 3) % len(ZEBRA)])) for j, (n, r) in enumerate(ZEBRA)]
    return list(range(RUNS)), fish


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    flocks, fish = tasks()
    if quick:
        flocks, fish = flocks[:2], fish[:1]
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run_flock, flocks)) + list(pool.map(_run_fish, fish))
    res = {"runs": runs}
    (out_dir / "e58_innovation_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e58_figures import make_figures
    make_figures(res, out_dir / "e58_figures", text)
    return text


if __name__ == "__main__":
    main()
