"""Experiment 44: per-fish speed map that depends on heading.

e43 (per-fish turning maps) left the virtual-group control unchanged: "none"
still died at stage 1 in 6 of 8 virtual groups. The exploratory diagnostic
e43b located the mechanism:
  * fitted social TURNING terms in virtual groups are ~0 (|coef| <= 0.005,
    real 0.08-0.19), so turning is clean;
  * the SPEED cohesion term is clearly positive in virtual groups
    (+0.25 to +0.53; real +1.2 to +2.7);
  * the "none" model's 1 s forecasts inflate spread by +1.2 to +2.1 cm in
    virtual groups (fish that never met), and k6-C partly removes this bias.
Interpretation: own speed depends on place AND heading (slow down when heading
into a wall, speed up into open water). e43's speed map had place only, so
"speed up when the others' centre is ahead" stood in for it.

Change (everything else as in e43): the per-fish speed regressors become
bias, v, dv_prev, and the 25 place bumps x (1, sin theta, cos theta), i.e.
78 features per fish; plus 2 wall terms (cos of the angle to the nearest
wall, at 5 and 15 cm), so speed can respond to heading into a wall.

Predictions, fixed before running:
  J1  The true rule survives stages 0-2 in T0-T4.
  J2  The answer has <= 3 candidates and contains the truth's neighbour rule
      with a subset of its terms (or "none" for T0), in T0-T4.
  J3  T5 (out of class): the micro-selected best candidate fails stage 2.
  J4  Mechanism: in the 6 virtual 8-fish groups, the "none" model's 1 s
      spread forecast bias is below 0.5 cm in >= 5 of 6 (e43b: 1.2-2.1).
  J5  Mechanism: the k6-C speed cohesion coefficient in virtual 8-fish groups
      is below 0.1 in absolute value in >= 5 of 6 (e43b: 0.25-0.53).
  J6  Virtual groups: "none" survives stages 0-1 in >= 6 of 8.
  J7  Gate open (paired virtual answer is exactly "none") in >= 5 of 8.
  J8  At least one candidate passes stage 2 in >= 4 of 8 real recordings.
Exploratory: gated rule claims on real fish; spread; search cost.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import experiments.e43_elimination_fish_maps as base
from experiments.e31_real_fish import ARENA, TRAIN, kinematics, virtual
from experiments.e39_multiscale_elimination import TRUTHS, macro
from experiments.e42_elimination_cohesion_gate import gate_open, pairs
from experiments.fish_data import RECORDINGS, load


def wall_speed(p, th):
    d = np.stack([p[..., 0], ARENA - p[..., 0], p[..., 1], ARENA - p[..., 1]], -1)
    normals = np.array([0.0, np.pi, np.pi / 2, -np.pi / 2])
    j = d.argmin(-1)
    dmin = np.take_along_axis(d, j[..., None], -1)[..., 0].clip(min=0.0)
    c = np.cos(normals[j] - th)
    return np.stack([c * np.exp(-dmin / 5.0), c * np.exp(-dmin / 15.0)], -1)


def own_speed_x(p, v, dv, th):
    b = base.bumps(p)
    return np.concatenate([np.stack([np.ones_like(v), v, dv], -1), wall_speed(p, th), b,
                           b * np.sin(th)[..., None], b * np.cos(th)[..., None]], -1)


def model_step(state, mdl, h, rng, gain=1.0):
    p, th, v, l1, l2, dv = state
    ct, bt, rt, cv, bv, rv = mdl
    turn = (base.own_turn_x(p, th, l1, l2) * ct).sum(-1) + rng.choice(rt, th.shape)
    dvn = (own_speed_x(p, v, dv, th) * cv).sum(-1) + rng.choice(rv, th.shape)
    if h != "none" and gain != 0.0:
        rule, ts = h.split("-")
        f = base.features(p, th, rule)
        turn = turn + gain * (f[..., base.TERMSETS[ts]] @ bt)
        if "C" in ts:
            dvn = dvn + gain * (f[..., base.SPEED_COLS] @ bv)
    v_new = np.clip(v + dvn, 0.0, base.V_MAX)
    th = (th + turn + np.pi) % (2 * np.pi) - np.pi
    p = np.clip(p + (v_new * base.DT)[..., None] * np.stack([np.cos(th), np.sin(th)], -1), 0.0, ARENA)
    return p, th, v_new, turn, l1, v_new - v


def fit_all(p, th, sp, stop, cands):
    l1, l2, ok_in, dvp = base.prepare(p, th, sp)
    own = base.own_turn_x(p, th, l1, l2)
    own_v = own_speed_x(p, sp, dvp, th)
    feats = {r: base.features(p, th, r) for r in base.RULES}
    okt = ok_in[1:].copy()
    okt[stop:] = False
    okv = np.zeros_like(okt)
    okv[:stop] = True
    pt, pv = base.PerFish(own[:-1], l1[1:], okt), base.PerFish(own_v[:-1], dvp[1:], okv)
    fits = {}
    for h in cands:
        zt, zv = base.social_turn(feats, h), base.social_speed(feats, h)
        ct, bt, rt = pt.fit(None if zt is None else zt[:-1])
        cv, bv, rv = pv.fit(None if zv is None else zv[:-1])
        fits[h] = (ct, bt, rt, cv, bv, rv)
    return fits, own, feats, pt, (l1, l2, ok_in, dvp)


base.model_step = model_step
base.fit_all = fit_all


def forecast_bias(p, th, sp):
    """Signed 1 s and 3 s forecast bias (pred - obs) of 'none' and k6-C on e43's stage-1 states."""
    T = len(th)
    cut = int(TRAIN * T)
    fits, _, _, _, (l1, l2, _, dvp) = fit_all(p, th, sp, cut, ("none", "k6-C"))
    idx = np.linspace(cut, T - max(base.HORIZONS) - 1, base.N_STATES).astype(int)
    out = {"k6C_speed_coef": fits["k6-C"][4].tolist(), "k6C_turn_coef": fits["k6-C"][1].tolist()}
    for h in ("none", "k6-C"):
        rng = np.random.default_rng(7)
        st = tuple(np.repeat(a[idx][:, None], base.REPLICAS, 1) for a in (p, th, sp, l1, l2, dvp))
        for s in range(1, max(base.HORIZONS) + 1):
            st = model_step(st, fits[h], h, rng)
            if s in base.HORIZONS:
                d = macro(st[0], st[1]).mean(1) - macro(p[idx + s], th[idx + s])
                out[f"{h}@{s}"] = d.mean(0).tolist()
    return out


def _synthetic(name):
    truth = TRUTHS[name]
    rng = np.random.default_rng(44_000 + int(name[1]))
    p, th, sp = kinematics(load(39, 0))
    sp = np.where(sp > 100.0, 0.0, sp)
    fits, _, _, _, _ = fit_all(p, th, sp, len(th), [truth])
    s0 = (p[0], th[0], np.clip(sp[0], 0.0, base.V_MAX), np.zeros(39), np.zeros(39), np.zeros(39))
    ps, ths = base.simulate(s0, fits[truth], truth, rng, base.BURN + base.SYNTH_STEPS,
                            switching=(name == "T5"))
    ps, ths = ps[base.BURN:], ths[base.BURN:]
    v = np.zeros(ths.shape)
    v[1:] = np.hypot(*np.moveaxis(ps[1:] - ps[:-1], -1, 0)) / base.DT
    v[0] = v[1]
    out = base.analyse(ps, ths, v, 44_100 + int(name[1]))
    out.update({"name": name, "truth": truth})
    return out


def _real(args):
    n, rec, kind = args
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    sp = np.where(sp > 100.0, 0.0, sp)
    out = base.analyse(p, th, sp, 44_500 + 10 * n + rec + (5 if kind == "virtual" else 0),
                       show_snap=(kind == "real"))
    if n == 8:
        out["bias"] = forecast_bias(p, th, sp)
    out.update({"N": n, "rec": rec, "kind": kind})
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    tasks = [(n, r, k) for n, recs in RECORDINGS.items() for r in recs for k in ("real", "virtual")]
    print(f"  {len(TRUTHS)} synthetic truths + {len(tasks)} real/virtual recordings ...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        synth = pool.map(_synthetic, TRUTHS)
        real = pool.map(_real, tasks)
        rows = {"synthetic": list(synth), "real": list(real)}
    (out_dir / "e44_elimination_speed_heading_raw.json").write_text(json.dumps(rows))
    text = analyze(rows)
    from experiments.e44_figures import make_figures
    make_figures(rows, out_dir / "e44_figures", text)
    return text


def predictions(rows):
    first = base.predictions(rows)[:3]
    pr = pairs(rows)
    real = [r for r, _ in pr]
    virt = [v for _, v in pr]
    v8 = [v for v in virt if v["N"] == 8]
    bias = [v["bias"]["none@10"][2] for v in v8]
    coef = [v["bias"]["k6C_speed_coef"][1] for v in v8]
    j4 = sum(abs(b) < 0.5 for b in bias)
    j5 = sum(abs(c) < 0.1 for c in coef)
    j6 = sum("none" in v["surv1"] for v in virt)
    j7 = sum(gate_open(v) for v in virt)
    j8 = sum(any(r["pass2"].values()) for r in real)
    return [(k.replace("H", "J"), *rest) for k, *rest in first] + [
        ("J4", "virtual 8-fish: 'none' 1 s spread bias < 0.5 cm (>= 5/6)", j4 >= 5,
         "bias " + ", ".join(f"{b:+.2f}" for b in bias)),
        ("J5", "virtual 8-fish: |k6-C speed cohesion| < 0.1 (>= 5/6)", j5 >= 5,
         "coef " + ", ".join(f"{c:+.2f}" for c in coef)),
        ("J6", "virtual: 'none' survives stages 0-1 (>= 6/8)", j6 >= 6, f"{j6}/8"),
        ("J7", "gate open: virtual answer is 'none' (>= 5/8)", j7 >= 5, f"{j7}/8"),
        ("J8", "a rule passes stage 2 (>= 4/8 real)", j8 >= 4, f"{j8}/8"),
    ]


def analyze(rows) -> str:
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        body = base.analyze(rows)
    body = body.split("\n  Pre-registered predictions:")[0]
    body = body.replace("Experiment 43: per-fish place maps (own model)",
                        "Experiment 44: per-fish speed maps with heading (on top of e43)")
    lines = [body, "", "  Forecast bias of 'none' (pred - obs, 1 s) and k6-C speed coefficients, 8 fish:",
             f"  {'':10}{'spread':>8}{'NND':>8}   k6-C speed (ahead, cohesion)"]
    for r in rows["real"]:
        if r["N"] == 8:
            b = r["bias"]["none@10"]
            c = r["bias"]["k6C_speed_coef"]
            lines.append(f"  {r['kind']:7} #{r['rec']}{b[2]:+8.2f}{b[3]:+8.2f}   {c[0]:+.3f} {c[1]:+.3f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(rows):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
