"""Experiment 43: multi-scale elimination with a per-fish, high-resolution place map.

e42 fixed stage 2 (equivalence margins) and cohesion (spread error 1.4-2.2x ->
1.05-1.36x), but the virtual-group gate stayed closed in 6 of 8 recordings:
time-shifted fish that never met were explained by attraction or cohesion
rules. Diagnosis: each fish's own behaviour was modelled with one shared,
quadratic place field. Fish that favour the same places then look as if they
steer toward each other, because "where the others are" is a proxy for "where
I like to go".

Change (everything else as in e42: 25 candidates, cohesion term, speed model,
adaptive stage 1, equivalence stage 2, gate, nested-minimal answers):
  Own model    per fish, not shared. Each fish i gets its own coefficients
               for: bias, 2 wall terms, 2 lagged turns, and a place map of
               25 Gaussian bumps (5 x 5 grid, width 20 cm) x (sin theta,
               cos theta), i.e. a preferred heading and strength anywhere in
               the tank (55 turn features). Speed: bias, v, dv_prev and the
               25 bumps (28 features), also per fish.
  Fitting      social coefficients stay shared across fish; own coefficients
               are fitted per fish (exact least squares by the
               Frisch-Waugh-Lovell partialling).
Synthetic truths T0-T5 are regenerated under this model (fitted on 39/0).

Predictions, fixed before running:
  H1  The true rule survives stages 0-2 in T0-T4.
  H2  The answer has <= 3 candidates and contains the truth's neighbour rule
      with a subset of its terms (or "none" for T0), in T0-T4.
  H3  T5 (out of class): the micro-selected best candidate fails stage 2.
  H4  Virtual groups: "none" survives stages 0-1 in >= 6 of 8 (e42: 2 of 8).
  H5  Gate open (paired virtual answer is exactly "none") in >= 5 of 8.
  H6  At least one candidate passes stage 2 in >= 4 of 8 real recordings.
  H7  Simulated spread (shown candidate) within 25% of observed in >= 6 of 8.
Exploratory: gated rule claims on real fish; search cost.

Figures (results/e43_figures/) are drawn from the raw results after the run.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import ARENA, TRAIN, kinematics, virtual, wall_terms
from experiments.e39_multiscale_elimination import (ANTI, CANDIDATES, RULES, SWITCH_P, TRUTHS,
                                                    block_z, macro, stage2_stats)
from experiments.e42_elimination_cohesion_gate import (SPEED_COLS, TERMSETS, features, gate_open,
                                                       nested_min, pairs, prepare, tolerance)
from experiments.fish_data import RECORDINGS, load

DT = 0.1
V_MAX = 60.0
N_STATES, REPLICAS, HORIZONS, BATCHES = 150, 4, (10, 30), 5
SIM_RUNS, SIM_STEPS, BURN = 2, 18000, 600
SYNTH_STEPS = 36000
Z0, Z1, Z1_EARLY = 3.0, 3.0, 4.0
SNAP = 30
GRID = np.linspace(10.0, 90.0, 5)
CENTRES = np.array([(x, y) for y in GRID for x in GRID])
WIDTH = 20.0
RIDGE = 1e-6


def bumps(p):
    d2 = ((p[..., None, :] - CENTRES) ** 2).sum(-1)
    return np.exp(-d2 / (2 * WIDTH ** 2))


def own_turn_x(p, th, l1, l2):
    b = bumps(p)
    return np.concatenate([np.ones(th.shape + (1,)), wall_terms(p, th), l1[..., None], l2[..., None],
                           b * np.sin(th)[..., None], b * np.cos(th)[..., None]], -1)


def own_speed_x(p, v, dv):
    return np.concatenate([np.stack([np.ones_like(v), v, dv], -1), bumps(p)], -1)


def social_turn(feats, h):
    if h == "none":
        return None
    rule, ts = h.split("-")
    return feats[rule][..., TERMSETS[ts]]


def social_speed(feats, h):
    if h == "none" or "C" not in h.split("-")[1]:
        return None
    return feats[h.split("-")[0]][..., SPEED_COLS]


class PerFish:
    """Per-fish own regressors with shared social coefficients (Frisch-Waugh-Lovell)."""

    def __init__(self, x, y, ok):
        self.x, self.y, self.ok = x, y, ok
        self.rows = [np.nonzero(ok[:, i])[0] for i in range(x.shape[1])]
        self.xi = [x[r, i] for i, r in enumerate(self.rows)]
        self.yi = [y[r, i] for i, r in enumerate(self.rows)]
        self.ginv = []
        for xi in self.xi:
            g = xi.T @ xi
            self.ginv.append(np.linalg.inv(g + RIDGE * np.trace(g) / len(g) * np.eye(len(g))))
        self.xty = [xi.T @ yi for xi, yi in zip(self.xi, self.yi)]

    def fit(self, z):
        n = len(self.rows)
        if z is None:
            b = np.zeros(0)
            zi = [None] * n
        else:
            zi = [z[r, i] for i, r in enumerate(self.rows)]
            k = z.shape[-1]
            a, c = np.zeros((k, k)), np.zeros(k)
            for x, y, zz, gi, xty in zip(self.xi, self.yi, zi, self.ginv, self.xty):
                xtz = x.T @ zz
                a += zz.T @ zz - xtz.T @ gi @ xtz
                c += zz.T @ y - xtz.T @ gi @ xty
            b = np.linalg.solve(a + 1e-9 * np.eye(k), c)
        coefs, resid = np.empty((n, self.x.shape[-1])), []
        for i, (x, y, zz, gi) in enumerate(zip(self.xi, self.yi, zi, self.ginv)):
            target = y if zz is None else y - zz @ b
            coefs[i] = gi @ (x.T @ target)
            resid.append(target - x @ coefs[i])
        return coefs, b, np.concatenate(resid)


def predict(own, coefs, z, b):
    out = (own * coefs).sum(-1)
    return out if z is None else out + z @ b


def model_step(state, mdl, h, rng, gain=1.0):
    p, th, v, l1, l2, dv = state
    ct, bt, rt, cv, bv, rv = mdl
    turn = (own_turn_x(p, th, l1, l2) * ct).sum(-1) + rng.choice(rt, th.shape)
    dvn = (own_speed_x(p, v, dv) * cv).sum(-1) + rng.choice(rv, th.shape)
    if h != "none" and gain != 0.0:
        rule, ts = h.split("-")
        f = features(p, th, rule)
        turn = turn + gain * (f[..., TERMSETS[ts]] @ bt)
        if "C" in ts:
            dvn = dvn + gain * (f[..., SPEED_COLS] @ bv)
    v_new = np.clip(v + dvn, 0.0, V_MAX)
    th = (th + turn + np.pi) % (2 * np.pi) - np.pi
    p = np.clip(p + (v_new * DT)[..., None] * np.stack([np.cos(th), np.sin(th)], -1), 0.0, ARENA)
    return p, th, v_new, turn, l1, v_new - v


def simulate(state, mdl, h, rng, steps, record_every=1, switching=False):
    mode, ps, ths = 1, [], []
    for t in range(steps):
        if switching and rng.random() < SWITCH_P:
            mode = 1 - mode
        if t % record_every == 0:
            ps.append(state[0]), ths.append(state[1])
        state = model_step(state, mdl, h, rng, gain=2.0 * mode if switching else 1.0)
    return np.array(ps), np.array(ths)


def fit_all(p, th, sp, stop, cands):
    l1, l2, ok_in, dvp = prepare(p, th, sp)
    own = own_turn_x(p, th, l1, l2)
    own_v = own_speed_x(p, sp, dvp)
    feats = {r: features(p, th, r) for r in RULES}
    okt = ok_in[1:].copy()
    okt[stop:] = False
    okv = np.zeros_like(okt)
    okv[:stop] = True
    pt, pv = PerFish(own[:-1], l1[1:], okt), PerFish(own_v[:-1], dvp[1:], okv)
    fits = {}
    for h in cands:
        zt, zv = social_turn(feats, h), social_speed(feats, h)
        ct, bt, rt = pt.fit(None if zt is None else zt[:-1])
        cv, bv, rv = pv.fit(None if zv is None else zv[:-1])
        fits[h] = (ct, bt, rt, cv, bv, rv)
    return fits, own, feats, pt, (l1, l2, ok_in, dvp)


def cohesion_coef(fits):
    out = {}
    for h in CANDIDATES:
        if h != "none" and "C" in h.split("-")[1]:
            ts = TERMSETS[h.split("-")[1]]
            out[h] = [float(fits[h][1][ts.index(4)]), float(fits[h][4][1])]
    return out


def analyse(p, th, sp, seed, show_snap=True):
    T, n = th.shape
    cut = int(TRAIN * T)
    t_start = time.perf_counter()
    fits, own, feats, pt, (l1, l2, ok_in, dvp) = fit_all(p, th, sp, cut, CANDIDATES)
    y, okf = l1[1:], ok_in[1:]
    shared = time.perf_counter() - t_start

    t0 = time.perf_counter()
    union = np.concatenate([feats[r][..., (0, 1, 2, 4)] for r in RULES], -1)[:-1]
    cu, _, _ = pt.fit(union)
    allv = okf.all(1)
    resid = (y - (own[:-1] * cu).sum(-1))[allv].sum(1)
    fsum = union[allv].sum(1)
    tr = np.nonzero(allv)[0] < cut
    xg = np.column_stack([np.ones(len(resid)), fsum])
    bg = np.linalg.lstsq(xg[tr], resid[tr], rcond=None)[0]
    e_full, e_twin = resid[~tr] - xg[~tr] @ bg, resid[~tr] - resid[tr].mean()
    z0 = block_z(e_twin ** 2 - e_full ** 2, 20)
    surv0 = [h for h in CANDIDATES if not (z0 > Z0 and h in ANTI)]
    time0 = time.perf_counter() - t0
    del union

    idx = np.linspace(cut, T - max(HORIZONS) - 1, N_STATES).astype(int)
    obs = np.concatenate([macro(p[idx + s], th[idx + s]) for s in HORIZONS], -1)
    err = {h: np.full((N_STATES, 2 * 4), np.nan) for h in surv0}
    time1 = {h: 0.0 for h in surv0}
    dropped_at, z1 = {}, {}
    alive, seen = list(surv0), np.zeros(N_STATES, bool)
    for b in range(BATCHES):
        bi = np.arange(b, N_STATES, BATCHES)
        for h in alive:
            t1 = time.perf_counter()
            rng = np.random.default_rng(seed * 100 + 10 * b + CANDIDATES.index(h))
            st = tuple(np.repeat(a[idx[bi]][:, None], REPLICAS, 1)
                       for a in (p, th, sp, l1, l2, dvp))
            pred = []
            for s in range(1, max(HORIZONS) + 1):
                st = model_step(st, fits[h], h, rng)
                if s in HORIZONS:
                    pred.append(macro(st[0], st[1]).mean(1))
            err[h][bi] = (obs[bi] - np.concatenate(pred, -1)) ** 2
            time1[h] += time.perf_counter() - t1
        seen[bi] = True
        if b == 0:
            continue
        si = np.nonzero(seen)[0]
        scale = np.median([err[h][si].mean(0) for h in alive], axis=0)
        tot = {h: (err[h][si] / scale).sum(1) for h in alive}
        best = min(alive, key=lambda h: tot[h].mean())
        thr = Z1 if b == BATCHES - 1 else Z1_EARLY
        for h in list(alive):
            z = block_z(tot[h] - tot[best], min(15, len(si) // 4))
            z1[h] = z
            if z > thr:
                alive.remove(h)
                dropped_at[h] = b
    surv1, best1 = alive, best

    test = okf.copy()
    test[:cut] = False
    step_err = {}
    for h in CANDIDATES:
        zt = social_turn(feats, h)
        pred = predict(own[:-1], fits[h][0], None if zt is None else zt[:-1], fits[h][1])
        step_err[h] = np.where(test, (y - pred) ** 2, 0.0)[cut:].sum(1)
    micro_best = min(CANDIDATES, key=lambda h: step_err[h].mean())
    del feats, own

    obs_series = macro(p[::10], th[::10])
    o = stage2_stats(obs_series, 10)
    o_mean, o_se = o.mean(0), o.std(0, ddof=1) / np.sqrt(10)
    tol = tolerance(o_mean)
    starts = (0, cut)
    show = best1 if surv1 else micro_best
    q2, time2, sim_means, snap = {}, {}, {}, {}
    for k, h in enumerate(CANDIDATES):
        t2 = time.perf_counter()
        rng = np.random.default_rng(seed * 1000 + k)
        blocks = []
        for run in range(SIM_RUNS):
            s = starts[run]
            s0 = (p[s], th[s], np.clip(sp[s], 0.0, V_MAX), np.zeros(n), np.zeros(n), np.zeros(n))
            ps, ths = simulate(s0, fits[h], h, rng, BURN + SIM_STEPS, record_every=10)
            ser = macro(ps[BURN // 10:], ths[BURN // 10:])
            blocks.append(stage2_stats(ser, 10 // SIM_RUNS))
            if show_snap and h == show and run == 0:
                snap = {"sim_p": ps[-SNAP:].tolist(), "sim_th": ths[-SNAP:].tolist(),
                        "sim_series": ser[::2].tolist()}
        sb = np.concatenate(blocks)
        s_mean, s_se = sb.mean(0), sb.std(0, ddof=1) / np.sqrt(len(sb))
        se = np.sqrt(o_se ** 2 + s_se ** 2)
        q2[h] = (np.abs(s_mean - o_mean) / (tol + 2 * se)).tolist()
        sim_means[h] = s_mean.tolist()
        time2[h] = time.perf_counter() - t2
    pass2 = {h: max(q2[h]) <= 1.0 for h in CANDIDATES}
    surv2 = [h for h in surv1 if pass2[h]]
    out = {"T": T, "n": n, "z0": z0, "surv0": surv0, "surv1": surv1,
           "z1": z1, "dropped_at": dropped_at, "micro_best": micro_best,
           "shared_s": shared, "stage0_s": time0, "stage1_s": time1,
           "cohesion_coef": cohesion_coef(fits), "place_map": fits["none"][0].tolist(),
           "q2": q2, "pass2": pass2, "surv2": surv2, "obs_stats": o_mean.tolist(),
           "obs_se": o_se.tolist(), "tol": tol.tolist(), "sim_stats": sim_means, "stage2_s": time2,
           "answer": nested_min(surv2), "abc_survivors": [h for h in CANDIDATES if pass2[h]],
           "shown": show, **snap,
           "cost_pruned": time0 + sum(time1.values()) + sum(time2[h] for h in surv1),
           "cost_abc": sum(time2.values())}
    if show_snap:
        obs_frames = np.arange(T - 10 * SNAP, T, 10)
        out.update({"obs_p": p[obs_frames].tolist(), "obs_th": th[obs_frames].tolist(),
                    "obs_series": obs_series[::2].tolist()})
    return out


def _synthetic(name):
    truth = TRUTHS[name]
    rng = np.random.default_rng(43_000 + int(name[1]))
    p, th, sp = kinematics(load(39, 0))
    sp = np.where(sp > 100.0, 0.0, sp)
    fits, _, _, _, _ = fit_all(p, th, sp, len(th), [truth])
    s0 = (p[0], th[0], np.clip(sp[0], 0.0, V_MAX), np.zeros(39), np.zeros(39), np.zeros(39))
    ps, ths = simulate(s0, fits[truth], truth, rng, BURN + SYNTH_STEPS, switching=(name == "T5"))
    ps, ths = ps[BURN:], ths[BURN:]
    v = np.zeros(ths.shape)
    v[1:] = np.hypot(*np.moveaxis(ps[1:] - ps[:-1], -1, 0)) / DT
    v[0] = v[1]
    out = analyse(ps, ths, v, 43_100 + int(name[1]))
    out.update({"name": name, "truth": truth})
    return out


def _real(args):
    n, rec, kind = args
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    out = analyse(p, th, np.where(sp > 100.0, 0.0, sp), 43_500 + 10 * n + rec + (5 if kind == "virtual" else 0),
                  show_snap=(kind == "real"))
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
    (out_dir / "e43_elimination_fish_maps_raw.json").write_text(json.dumps(rows))
    text = analyze(rows)
    from experiments.e43_figures import make_figures
    make_figures(rows, out_dir / "e43_figures", text)
    return text


def predictions(rows):
    syn = {r["name"]: r for r in rows["synthetic"]}
    pr = pairs(rows)
    real = [r for r, _ in pr]
    virt = [v for _, v in pr]
    in_class = [syn[t] for t in ("T0", "T1", "T2", "T3", "T4")]

    def answer_ok(r):
        if len(r["answer"]) > 3:
            return False
        if r["truth"] == "none":
            return "none" in r["answer"]
        rt, tt = r["truth"].split("-")
        return any(h != "none" and h.split("-")[0] == rt and set(h.split("-")[1]) <= set(tt)
                   for h in r["answer"])

    t5 = syn["T5"]
    h4 = sum("none" in v["surv1"] for v in virt)
    h5 = sum(gate_open(v) for v in virt)
    h6 = sum(any(r["pass2"].values()) for r in real)
    sp = [r["sim_stats"][r["shown"]][3] / r["obs_stats"][3] for r in real]
    h7 = sum(abs(x - 1) <= 0.25 for x in sp)
    return [
        ("H1", "truth survives stages 0-2 (T0-T4)", all(r["truth"] in r["surv2"] for r in in_class),
         ", ".join(f"{r['name']}:{'yes' if r['truth'] in r['surv2'] else 'no'}" for r in in_class)),
        ("H2", "answer <= 3, contains truth's rule (T0-T4)", all(answer_ok(r) for r in in_class),
         ", ".join(f"{r['name']}:{'+'.join(r['answer']) or '-'}" for r in in_class)),
        ("H3", "T5: micro-best fails stage 2", not t5["pass2"][t5["micro_best"]],
         f"{t5['micro_best']}, max q {max(t5['q2'][t5['micro_best']]):.2f}"),
        ("H4", "virtual: 'none' survives stages 0-1 (>= 6/8)", h4 >= 6, f"{h4}/8"),
        ("H5", "gate open: virtual answer is 'none' (>= 5/8)", h5 >= 5, f"{h5}/8"),
        ("H6", "a rule passes stage 2 (>= 4/8 real)", h6 >= 4, f"{h6}/8"),
        ("H7", "simulated spread within 25% (>= 6/8 real)", h7 >= 6,
         "sim/obs " + ", ".join(f"{x:.2f}" for x in sp)),
    ]


def analyze(rows) -> str:
    syn = rows["synthetic"]
    pr = pairs(rows)
    lines = ["Experiment 43: per-fish place maps (own model), with e42's cohesion, tolerance and gate", "",
             "  Part A, synthetic schools with a known rule (39 fish):",
             f"  {'':4}{'truth':>8}{'z0':>7}{'left 0':>8}{'left 1':>8}{'left 2':>8}{'ABC':>5}"
             f"{'cost':>7}  answer | micro best"]
    for r in syn:
        lines.append(f"  {r['name']:4}{r['truth']:>8}{r['z0']:7.1f}{len(r['surv0']):8d}{len(r['surv1']):8d}"
                     f"{len(r['surv2']):8d}{len(r['abc_survivors']):5d}"
                     f"{r['cost_pruned'] / r['cost_abc']:7.2f}  {', '.join(r['answer']) or '(none)'}"
                     f" | {r['micro_best']}")
    lines += ["", "  Part B, real schools (Puy et al. 2024) and their virtual twins:",
              f"  {'N':>3}{'rec':>4}{'z0':>7}{'left 1':>8}{'left 2':>8}  real answer"
              f"{'':14}| virtual: z0, left 1, answer | gate"]
    for r, v in pr:
        lines.append(f"  {r['N']:3d}{r['rec']:4d}{r['z0']:7.1f}{len(r['surv1']):8d}{len(r['surv2']):8d}  "
                     f"{', '.join(r['answer']) or '(space eliminated)':25}| {v['z0']:5.1f} "
                     f"{len(v['surv1']):3d}  {', '.join(v['answer']) or '(space eliminated)':12}| "
                     f"{'OPEN' if gate_open(v) else 'closed'}")
    lines += ["", "  Real schools, observed vs shown candidate (polarization mean, SD, milling, "
              "spread cm, NND cm):"]
    for r, _ in pr:
        lines.append(f"  {r['N']:3d}{r['rec']:4d}  obs " + " ".join(f"{x:6.2f}" for x in r["obs_stats"])
                     + f"   {r['shown']:>7} " + " ".join(f"{x:6.2f}" for x in r["sim_stats"][r["shown"]]))
    claims = [(r, v) for r, v in pr if gate_open(v) and r["answer"]]
    lines += ["", "  Gated rule claims on real fish (exploratory): "
              + ("; ".join(f"{r['N']} fish #{r['rec']}: {', '.join(r['answer'])}" for r, _ in claims)
                 or "none")]
    inc = [r for r in syn if r["name"] != "T5"]
    lines.append(f"  Search cost ratio, T0-T4 (exploratory): "
                 f"{sum(r['cost_pruned'] for r in inc) / sum(r['cost_abc'] for r in inc):.2f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(rows):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
