"""Experiment 53: does the influence map replicate in other species?

e52 found, in eight real schools of one fish species (Puy et al. 2024), that
the school's next turn is carried for about a second by fast individuals at
the front heading off the school's line, and that virtual groups show none
of it. e53 runs e52's method unchanged on two independent data sets:

  zebrafish  Heras et al. 2019: nine 10-minute recordings of 60, 80 and 100
             juvenile zebrafish in a circular arena (loader of e38; every 3rd
             frame, about 0.094 s; body lengths).
  pigeons    Nagy et al. 2013: free flights of three 10-bird flocks (A, B, C;
             4-5 flights each, 19-48 minutes of flight per flock; loader of
             e35-e37 at 0.2 s; samples where every bird flies > 5 m/s;
             flights concatenated with the joins excluded).

Method (e52's, with only the unit changes these data need):
  horizon h = 1 s (3 s exploratory) and history LAGS = 5 steps, converted
  to steps of each data set; rows whose window [t - LAGS - 1, t + h] crosses
  a break (flight join, virtual seam) are excluded; first 70% fitted, last
  30% measured; e51's cross-validated ridge with extra-feature scale, block
  bootstrap intervals; target = the group's heading change over h.
  Block = the group state Z (lags 0-4, quadratic terms), the group's last 5
  turning rates, and a place map of the centroid (5 x 5 bumps over the
  training range of the centroid, times 1, sin and cos of the group heading),
  replacing e50's tank-specific map.
  conf = e50's configuration (computed in time chunks for large groups);
  infl = e52's 75 mirror-symmetric influence features.
  Virtual groups: each individual's own track (kinematics computed first)
  shifted in time by i T / N within its recording or flight; seams excluded.
  Null: the influence features of a group of another size (zebrafish) or
  another flock (pigeons), same kind, tiled; calibration: e52's planted
  5% signal. Bootstrap blocks are 300 rows or n/4 if shorter (pigeon flock
  B's virtual group keeps only about 340 test rows at 1 s after its seams);
  no interval below 50 rows.

Predictions, fixed before running (1 s horizon):
  R0  The test works on both data sets: planted signal raises G by >= 0.025
      with lower bound > 0, and null |G| <= 0.01, each in >= 20 of the 24
      groups (9 + 9 zebrafish, 3 + 3 pigeon).
  Z1  The learned map informs the zebrafish group's turn: G(infl) >= 0.03
      with lower bound > 0 in >= 7 of 9 real groups.
  Z2  Interaction, not persistence: G(infl) real > virtual in >= 7 of 9 pairs.
  Z3  Front fish lead: the sin-phi map is > 0 at the front (x = 1, y = 0) and
      larger there than at the back (x = -1, y = 0) in >= 7 of 9 real groups.
  Z4  Speed carries the most: G(speed channels alone) >= G(heading channels
      alone) in >= 7 of 9 real groups.
  P1-P4  The same four for the pigeon flocks, in >= 2 of 3.
Exploratory: 3 s; conf against infl; all channel groups; the maps.
"""

from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e49_school_layer import LAGS, school_state
from experiments.e50_school_configuration import _wrap, configuration
from experiments.e51_trusted_closure import BLOCK, BOOT, PLANT, Ridge, _tame
from experiments.e52_influence_map import GRID, GROUPS, MapRidge, _density, bumps, channel_maps, influence

SEED = 53
SECONDS = (1, 3)
ZEBRA = [(n, r) for n in (60, 80, 100) for r in (1, 2, 3)]
FLOCKS = ("A", "B", "C")


def zebrafish(n, rec, kind):
    from experiments.e38_zebrafish import kinematics, load
    pos, _, dt = load(n, rec)
    p, th, sp = kinematics(pos, dt)
    brk = np.zeros(len(th), bool)
    brk[0] = True
    if kind == "virtual":
        p, th, sp, brk = _shift(p, th, sp, brk)
    return p, th, sp, brk, dt


def pigeons(flock, kind):
    from experiments.pigeon_data import DT, flights, load
    parts = []
    for f in flights(flock):
        s = load(f, 2)
        brk = ~s["cont"]
        brk[0] = True
        part = (s["p"], s["theta"], s["speed"], brk)
        parts.append(_shift(*part) if kind == "virtual" else part)
    return (*(np.concatenate([q[k] for q in parts]) for k in range(4)), DT)


def _shift(p, th, sp, brk):
    """Each individual's track shifted by i T / N; the seams become breaks."""
    T, n = th.shape
    p, th, sp, out = p.copy(), th.copy(), sp.copy(), brk.copy()
    for i in range(1, n):
        k = i * T // n
        p[:, i], th[:, i], sp[:, i] = np.roll(p[:, i], k, 0), np.roll(th[:, i], k, 0), np.roll(sp[:, i], k, 0)
        out |= np.roll(brk, k)
        out[k] = True
    return p, th, sp, out


def conf_chunked(p, th, sp, chunk=400):
    parts = [configuration(p[i:i + chunk], th[i:i + chunk], sp[i:i + chunk]) for i in range(0, len(th), chunk)]
    return (np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts]),
            np.concatenate([q[2] for q in parts]))


def place_map(z, ref):
    lo, hi = np.percentile(z[ref, :2], [2, 98], axis=0)
    u = (z[:, :2] - lo) / np.maximum(hi - lo, 1e-9)
    c = np.linspace(0, 1, 5)
    cx, cy = np.meshgrid(c, c)
    b = np.exp(-((u[:, None, 0] - cx.ravel()) ** 2 + (u[:, None, 1] - cy.ravel()) ** 2) / (2 * 0.2 ** 2))
    ang = np.arctan2(z[:, 3], z[:, 2])
    return np.concatenate([b, b * np.sin(ang)[:, None], b * np.cos(ang)[:, None]], 1)


def gap_ci(q_extra, q_block, rng):
    """e51's G with a moving-block bootstrap; blocks shrink to n/4 for short test sets (none below 50 rows)."""
    n = len(q_block)
    g = float(1 - q_extra.sum() / q_block.sum())
    if n < 50:
        return {"G": g, "lo": float("nan"), "hi": float("nan")}
    block = min(BLOCK, n // 4)
    starts = rng.integers(0, n - block + 1, size=(BOOT, n // block))
    idx = (starts[..., None] + np.arange(block)).reshape(BOOT, -1)
    lo, hi = np.percentile(1 - q_extra[idx].sum(1) / q_block[idx].sum(1), [2.5, 97.5])
    return {"G": g, "lo": float(lo), "hi": float(hi)}


def _rows(brk, T, h):
    bad = np.zeros(T, bool)
    for s in np.nonzero(brk)[0]:
        bad[max(0, s - h - 1):s + LAGS + 2] = True
    rows = np.arange(LAGS + 1, T - h)
    rows = rows[~bad[rows]]
    cut = int(0.7 * T)
    train, test = rows[rows + h < cut], rows[rows >= cut]
    return train, test


def _data(spec, kind):
    if spec[0] == "zebrafish":
        return zebrafish(spec[1], spec[2], kind)
    return pigeons(spec[1], kind)


def _run(args):
    spec, kind, null_spec = args
    rng = np.random.default_rng(SEED * 1000 + zlib.crc32(repr((spec, kind)).encode()) % 997)
    p, th, sp, brk, dt = _data(spec, kind)
    T = len(th)
    train_all = np.arange(int(0.7 * T))
    z = school_state(p, th, sp)
    conf, lead, big_th = conf_chunked(p, th, sp)
    conf = _tame(np.column_stack([conf, lead]), train_all)
    infl, fx, fy = influence(p, th, sp, train_all)
    po, tho, spo, _, _ = _data(null_spec, kind)
    null, _, _ = influence(po, tho, spo, np.arange(len(tho)))
    null = np.resize(null, infl.shape)
    rates = np.column_stack([np.roll(_wrap(big_th - np.roll(big_th, 1)), j) for j in range(5)])
    iu = np.triu_indices(z.shape[1])
    lagged = np.concatenate([np.roll(z, j, axis=0) for j in range(LAGS)], 1)
    quad = (z[:, :, None] * z[:, None, :])[:, iu[0], iu[1]]
    block = np.concatenate([lagged, quad, place_map(z, train_all), rates], 1)
    with_conf = np.concatenate([block, conf, conf ** 2], 1)
    nb = block.shape[1]
    names = bumps()
    feats = {"conf": with_conf, "infl": np.concatenate([block, infl], 1),
             "null": np.concatenate([block, null], 1)}
    for g, chans in GROUPS.items():
        idx = [j for j, (c, *_) in enumerate(names) if c in chans]
        feats["only_" + g] = np.concatenate([block, infl[:, idx]], 1)
    ones_block = np.column_stack([np.ones(T), block])
    j0 = names.index(("sin", -1, 0.75, 0.0))

    out = {"data": spec[0], "group": str(spec[1:]), "N": int(th.shape[1]), "kind": kind, "dt": float(dt),
           "null": str(null_spec[1:]), "density": _density(fx, fy, train_all), "h": {}}
    for sec in SECONDS:
        h = int(round(sec / dt))
        train, test = _rows(brk, T, h)
        rows = np.concatenate([train, test])
        y = _wrap(big_th[rows + h] - big_th[rows])
        pos_tr, pos_te = np.arange(len(train)), np.arange(len(train), len(rows))
        y_tr, y_te, one = y[pos_tr, None], y[pos_te], np.ones(1)
        base = Ridge(block, train, test, h + LAGS)
        pb = base.fit(y_tr, one)[0][:, 0]
        res = {"h_steps": h, "n_train": len(train), "n_test": len(test)}
        for k, x in feats.items():
            m = MapRidge(x, train, test, h + LAGS, nb)
            pr, lam, s, coef = m.fit_coef(y_tr, one)
            res[k] = gap_ci((y_te - pr[:, 0]) ** 2, (y_te - pb) ** 2, rng) | {"lam": lam, "scale": s}
            if k == "infl":
                pm = m
                if sec == 1:
                    maps = channel_maps(coef[nb:, 0])
                    out["maps"] = {c: v.tolist() for c, v in maps.items()}
                    i_f, i_b, i_0 = (int(np.argmin(np.abs(GRID - v))) for v in (1, -1, 0))
                    out["front_back"] = [float(maps["sin"][i_0, i_f]), float(maps["sin"][i_0, i_b])]
        xi = infl[:, j0] - ones_block @ np.linalg.lstsq(ones_block[rows], infl[rows, j0], rcond=None)[0]
        kappa = np.sqrt(PLANT / (1 - PLANT) * ((y_te - pb) ** 2).mean() / xi[test].var())
        yp = y + kappa * xi[rows]
        fb = base.fit(yp[pos_tr, None], one)[0][:, 0]
        fi = pm.fit(yp[pos_tr, None], one)[0][:, 0]
        res["planted"] = {"before": res["infl"], "after": gap_ci((yp[pos_te] - fi) ** 2, (yp[pos_te] - fb) ** 2, rng)}
        out["h"][str(sec)] = res
    return out


def _sel(res, data, kind):
    return [r for r in res["runs"] if r["data"] == data and r["kind"] == kind]


def predictions(res):
    allg = [r["h"]["1"] for r in res["runs"]]
    cal = sum(g["planted"]["after"]["G"] - g["planted"]["before"]["G"] >= 0.025 and g["planted"]["after"]["lo"] > 0
              for g in allg)
    nul = sum(abs(g["null"]["G"]) <= 0.01 for g in allg)
    out = [("R0", f"the test works: planted rise >= 0.025, lower > 0 and null |G| <= 0.01 (each >= 20/{len(allg)})",
            cal >= 20 and nul >= 20, f"planted {cal}/{len(allg)}, null {nul}/{len(allg)}")]
    for data, key, need in (("zebrafish", "Z", 7), ("pigeons", "P", 2)):
        real, virt = _sel(res, data, "real"), _sel(res, data, "virtual")
        n = len(real)
        rg = [r["h"]["1"] for r in real]
        c1 = sum(g["infl"]["G"] >= 0.03 and g["infl"]["lo"] > 0 for g in rg)
        c2 = sum(r["h"]["1"]["infl"]["G"] > v["h"]["1"]["infl"]["G"] for r, v in zip(real, virt))
        c3 = sum(f > 0 and f > b for f, b in (r["front_back"] for r in real))
        c4 = sum(g["only_speed"]["G"] >= g["only_heading"]["G"] for g in rg)
        fmt = lambda gs, k: ", ".join(f"{g[k]['G']:+.3f}" for g in gs)  # noqa: E731
        out += [
            (f"{key}1", f"{data}: learned map informs the turn, G >= 0.03, lower > 0 (>= {need}/{n})", c1 >= need,
             f"{c1}/{n}; {fmt(rg, 'infl')}"),
            (f"{key}2", f"{data}: real > virtual (>= {need}/{n})", c2 >= need,
             f"{c2}/{n}; virtual {fmt([v['h']['1'] for v in virt], 'infl')}"),
            (f"{key}3", f"{data}: front leads, sin-phi map front > 0 and > back (>= {need}/{n})", c3 >= need,
             f"{c3}/{n}; " + ", ".join(f"{f:+.2f}/{b:+.2f}" for f, b in (r['front_back'] for r in real))),
            (f"{key}4", f"{data}: speed channels >= heading channels (>= {need}/{n})", c4 >= need,
             f"{c4}/{n}; speed {fmt(rg, 'only_speed')} vs heading {fmt(rg, 'only_heading')}"),
        ]
    return out


def analyze(res) -> str:
    lines = ["Experiment 53: does the influence map replicate in zebrafish and pigeons?", ""]
    for r in res["runs"]:
        lines.append(f"  {r['data']:9} {r['group']:10} N={r['N']:3d} {r['kind']:7} (dt {r['dt']:.3f} s, null {r['null']})")
        for sec in SECONDS:
            g = r["h"][str(sec)]
            cell = lambda k: f"{g[k]['G']:+.3f} [{g[k]['lo']:+.3f},{g[k]['hi']:+.3f}]"  # noqa: E731
            lines.append(f"    {sec} s ({g['h_steps']} steps, {g['n_train']}/{g['n_test']} rows)  conf {cell('conf')}"
                         f"  infl {cell('infl')}  null {g['null']['G']:+.3f}  planted {g['planted']['before']['G']:+.3f}"
                         f" -> {g['planted']['after']['G']:+.3f}")
            lines.append("          one channel group: " + "  ".join(f"{k} {g['only_' + k]['G']:+.3f}" for k in GROUPS))
        f, b = r["front_back"]
        lines.append(f"          sin-phi map: front {f:+.4f}, back {b:+.4f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def tasks():
    out = []
    for i, (n, r) in enumerate(ZEBRA):
        null = ZEBRA[(i + 3) % len(ZEBRA)]
        out += [(("zebrafish", n, r), k, ("zebrafish", *null)) for k in ("real", "virtual")]
    for i, f in enumerate(FLOCKS):
        out += [(("pigeons", f), k, ("pigeons", FLOCKS[(i + 1) % 3])) for k in ("real", "virtual")]
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    jobs = tasks()
    if quick:
        jobs = [j for j in jobs if j[0] in (("zebrafish", 60, 1), ("pigeons", "B"))]
    with ProcessPoolExecutor(max_workers=6) as pool:
        runs = list(pool.map(_run, jobs))
    res = {"runs": runs, "seconds": list(SECONDS)}
    (out_dir / "e53_influence_replication_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e53_figures import make_figures
    make_figures(res, out_dir / "e53_figures", text)
    return text


if __name__ == "__main__":
    main()
