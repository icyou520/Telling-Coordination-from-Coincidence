"""Experiment 42: multi-scale elimination with an equivalence-margin stage 2,
a distance-growing cohesion term, and the virtual-group gate.

e41 diagnosed three problems:
  1. Stage 2 rejected the true rule in T0-T2 on mismatches of 2-13%: 30 min
     of simulation makes the block standard error tiny, so a z-test demands
     an exact match that a refitted model never gives.
  2. No cohesion: simulated 8-fish schools 1.6-2.2x too spread out, despite
     speed dynamics.
  3. Virtual groups (no interaction) still lose "none" at stage 1 (7 of 8).

Changes (everything else as in e41: 25 candidates, place field, speed model,
adaptive stage 1, nested-minimal answers, thresholds Z0 = 3, Z1 = 3/4):
  Stage 2      equivalence test. A candidate passes if, for every statistic,
               |sim - obs| <= tol + 2 * SE, with tol = 0.05 (absolute) for
               polarization mean, polarization SD and milling, and 15% of
               the observed value for spread and nearest-neighbour distance.
               (Margins chosen from e41's truth mismatches, before e42's
               truths were generated; T5 still differs by 0.09 in
               polarization SD.)
  Cohesion     attraction now has a second, distance-growing term. With
               c_i the mean position of i's neighbours under the rule, d_i
               the distance to it (capped at 30 cm) and beta_i its bearing:
                 turn  += g_t * sin(beta_i - theta_i) * d_i / 10
                 speed += g_v * cos(beta_i - theta_i) * d_i / 10
               Termsets: C = (sum attraction, cohesion), AC, ACR likewise.
               Synthetic truths T0-T5 are regenerated under this model.
  Gate         virtual groups now run all stages. A real recording's rule
               is claimed only if its paired virtual group's answer is
               exactly "none" (the no-interaction model survives and nothing
               simpler-than-everything else is needed).

Predictions, fixed before running:
  G1  The true rule survives stages 0-2 in T0-T4.
  G2  The answer has <= 3 candidates and contains the truth's neighbour rule
      with a subset of its terms (or "none" for T0), in T0-T4.
  G3  T5 (out of class): the micro-selected best candidate fails stage 2.
  G4  Cohesion is used: the k6-AC turning cohesion coefficient g_t > 0 in
      >= 7 of 8 real recordings.
  G5  Cohesion works: the shown candidate's simulated spread is within 25%
      of the observed spread in >= 6 of 8 real recordings.
  G6  At least one candidate passes stage 2 in >= 4 of 8 real recordings.
  G7  Virtual groups: "none" passes stage 2 in >= 7 of 8.
  G8  Gate open (paired virtual answer is exactly "none") in >= 4 of 8.
  G9  Pruned search cost <= 0.5 x ABC-all search cost, summed over T0-T4.
Exploratory: the rules claimed on real fish where the gate is open.

Figures (results/e42_figures/) are drawn from the raw results after the run.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import ARENA, TRAIN, kinematics, virtual, wall_terms
from experiments.e32_fish_closed_twin import turns
from experiments.e39_multiscale_elimination import (ANTI, CANDIDATES, RULES, SWITCH_P, TRUTHS,
                                                    block_z, macro, stage2_stats)
from experiments.fish_data import RECORDINGS, load

DT = 0.1
N_OWN, N_OWN_V, V_MAX = 17, 3, 60.0
N_STATES, REPLICAS, HORIZONS, BATCHES = 150, 4, (10, 30), 5
SIM_RUNS, SIM_STEPS, BURN = 2, 18000, 600
SYNTH_STEPS = 36000
Z0, Z1, Z1_EARLY = 3.0, 3.0, 4.0
TOL_ABS, TOL_REL = (0.05, 0.05, 0.05, None, None), (None, None, None, 0.15, 0.15)
D_CAP = 30.0
SNAP = 30
TERMSETS = {"A": (0,), "C": (1, 4), "AC": (0, 1, 4), "ACR": (0, 1, 2, 4)}
SPEED_COLS = (3, 5)


def features(p, theta, rule, chunk=400):
    """Neighbour terms: alignment, attraction, repulsion, 'ahead' (speed),
    cohesion (turn), cohesion (speed)."""
    lead, n = theta.shape[:-1], theta.shape[-1]
    p2, th2 = p.reshape(-1, n, 2), theta.reshape(-1, n)
    out = np.empty((len(th2), n, 6))
    eye = np.eye(n, dtype=bool)
    for a in range(0, len(th2), chunk):
        pp, th = p2[a:a + chunk], th2[a:a + chunk]
        diff = pp[:, None, :, :] - pp[:, :, None, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        dist[:, eye] = np.inf
        if rule[0] == "k":
            k = min(int(rule[1:]), n - 1)
            w = np.zeros_like(dist)
            np.put_along_axis(w, np.argpartition(dist, k - 1, axis=-1)[..., :k], 1.0, axis=-1)
        else:
            w = (dist < float(rule[1:])).astype(float)
        rel = np.arctan2(diff[..., 1], diff[..., 0]) - th[:, :, None]
        sb = np.sin(rel) * w
        out[a:a + chunk, :, 0] = (np.sin(th[:, None, :] - th[:, :, None]) * w).sum(-1)
        out[a:a + chunk, :, 1] = sb.sum(-1)
        out[a:a + chunk, :, 2] = (sb * np.exp(-dist / 2.0)).sum(-1)
        out[a:a + chunk, :, 3] = (np.cos(rel) * w).sum(-1)
        cnt = w.sum(-1)
        vec = (diff * w[..., None]).sum(-2) / np.maximum(cnt, 1.0)[..., None]
        dc = np.minimum(np.hypot(vec[..., 0], vec[..., 1]), D_CAP) / 10.0
        beta = np.arctan2(vec[..., 1], vec[..., 0]) - th
        has = cnt > 0
        out[a:a + chunk, :, 4] = np.where(has, np.sin(beta) * dc, 0.0)
        out[a:a + chunk, :, 5] = np.where(has, np.cos(beta) * dc, 0.0)
    return out.reshape(lead + (n, 6))


def own_turn_x(p, th, l1, l2):
    x = (p[..., 0] - ARENA / 2) / (ARENA / 2)
    y = (p[..., 1] - ARENA / 2) / (ARENA / 2)
    basis = (np.ones_like(x), x, y, x * x, y * y, x * y)
    s, c = np.sin(th), np.cos(th)
    return np.concatenate([np.ones(th.shape + (1,)), wall_terms(p, th), l1[..., None],
                           l2[..., None], np.stack([b * s for b in basis] + [b * c for b in basis], -1)],
                          -1)


def own_speed_x(v, dv):
    return np.stack([np.ones_like(v), v, dv], -1)


def turn_x(own, feats, h):
    if h == "none":
        return own
    rule, ts = h.split("-")
    return np.concatenate([own, feats[rule][..., TERMSETS[ts]]], -1)


def speed_x(own_v, feats, h):
    if h == "none" or "C" not in h.split("-")[1]:
        return own_v
    return np.concatenate([own_v, feats[h.split("-")[0]][..., SPEED_COLS]], -1)


def model_step(state, mdl, h, rng, gain=1.0):
    p, th, v, l1, l2, dv = state
    ct, rt, cv, rv = mdl
    turn = own_turn_x(p, th, l1, l2) @ ct[:N_OWN] + rng.choice(rt, th.shape)
    dvn = own_speed_x(v, dv) @ cv[:N_OWN_V] + rng.choice(rv, th.shape)
    if h != "none" and gain != 0.0:
        rule, ts = h.split("-")
        f = features(p, th, rule)
        turn = turn + gain * (f[..., TERMSETS[ts]] @ ct[N_OWN:])
        if "C" in ts:
            dvn = dvn + gain * (f[..., SPEED_COLS] @ cv[N_OWN_V:])
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


def nested_min(surv):
    def simpler(a, b):
        if a == b:
            return False
        if a == "none":
            return True
        if b == "none":
            return False
        (ra, ta), (rb, tb) = a.split("-"), b.split("-")
        return ra == rb and set(ta) < set(tb)
    return [h for h in surv if not any(simpler(g, h) for g in surv)]


def tolerance(o_mean):
    return np.array([a if a is not None else r * abs(o) for a, r, o in zip(TOL_ABS, TOL_REL, o_mean)])


def prepare(p, th, sp):
    l1, l2, ok_in = turns(th, sp)
    dvp = np.zeros_like(sp)
    dvp[1:] = sp[1:] - sp[:-1]
    return l1, l2, ok_in, dvp


def fit_all(p, th, sp, stop, cands):
    l1, l2, ok_in, dvp = prepare(p, th, sp)
    own = own_turn_x(p, th, l1, l2)
    own_v = own_speed_x(sp, dvp)
    feats = {r: features(p, th, r) for r in RULES}
    yt, yv = l1[1:], dvp[1:]
    okt = ok_in[1:].copy()
    okt[stop:] = False
    okv = np.zeros_like(okt)
    okv[:stop] = True
    fits = {}
    for h in cands:
        xt, xv = turn_x(own, feats, h)[:-1], speed_x(own_v, feats, h)[:-1]
        ct = np.linalg.lstsq(xt[okt], yt[okt], rcond=None)[0]
        cv = np.linalg.lstsq(xv[okv], yv[okv], rcond=None)[0]
        fits[h] = (ct, yt[okt] - xt[okt] @ ct, cv, yv[okv] - xv[okv] @ cv)
    return fits, own, feats, (l1, l2, ok_in, dvp)


def cohesion_coef(fits):
    out = {}
    for h in CANDIDATES:
        if h != "none" and "C" in h.split("-")[1]:
            ts = TERMSETS[h.split("-")[1]]
            out[h] = [float(fits[h][0][N_OWN + ts.index(4)]), float(fits[h][2][N_OWN_V + 1])]
    return out


def analyse(p, th, sp, seed, show_snap=True):
    T, n = th.shape
    cut = int(TRAIN * T)
    t_start = time.perf_counter()
    fits, own, feats, (l1, l2, ok_in, dvp) = fit_all(p, th, sp, cut, CANDIDATES)
    y, okf = l1[1:], ok_in[1:]
    train = okf.copy()
    train[cut:] = False
    shared = time.perf_counter() - t_start

    t0 = time.perf_counter()
    union = np.concatenate([own] + [feats[r][..., (0, 1, 2, 4)] for r in RULES], -1)[:-1]
    cu = np.linalg.lstsq(union[train], y[train], rcond=None)[0]
    allv = okf.all(1)
    resid = (y - own[:-1] @ cu[:N_OWN])[allv].sum(1)
    fsum = union[allv][..., N_OWN:].sum(1)
    tr = np.nonzero(allv)[0] < cut
    xg = np.column_stack([np.ones(len(resid)), fsum])
    bg = np.linalg.lstsq(xg[tr], resid[tr], rcond=None)[0]
    e_full, e_twin = resid[~tr] - xg[~tr] @ bg, resid[~tr] - resid[tr].mean()
    z0 = block_z(e_twin ** 2 - e_full ** 2, 20)
    s_omega = float(1 - np.mean(e_full ** 2) / np.mean(e_twin ** 2))
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
    step_err = {h: np.where(test, (y - turn_x(own, feats, h)[:-1] @ fits[h][0]) ** 2, 0.0)[cut:].sum(1)
                for h in CANDIDATES}
    micro_best = min(CANDIDATES, key=lambda h: step_err[h].mean())
    del feats, own

    obs_series = macro(p[::10], th[::10])
    o = stage2_stats(obs_series, 10)
    o_mean, o_se = o.mean(0), o.std(0, ddof=1) / np.sqrt(10)
    tol = tolerance(o_mean)
    starts = (0, cut)
    show = best1 if surv1 else micro_best
    q2, z2, time2, sim_means, snap = {}, {}, {}, {}, {}
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
        z2[h] = (np.abs(s_mean - o_mean) / se).tolist()
        sim_means[h] = s_mean.tolist()
        time2[h] = time.perf_counter() - t2
    pass2 = {h: max(q2[h]) <= 1.0 for h in CANDIDATES}
    surv2 = [h for h in surv1 if pass2[h]]
    out = {"T": T, "n": n, "z0": z0, "S_omega": s_omega, "surv0": surv0, "surv1": surv1,
           "z1": z1, "dropped_at": dropped_at, "micro_best": micro_best,
           "shared_s": shared, "stage0_s": time0, "stage1_s": time1,
           "cohesion_coef": cohesion_coef(fits),
           "q2": q2, "z2": z2, "pass2": pass2, "surv2": surv2, "obs_stats": o_mean.tolist(),
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
    rng = np.random.default_rng(42_000 + int(name[1]))
    p, th, sp = kinematics(load(39, 0))
    sp = np.where(sp > 100.0, 0.0, sp)
    fits, _, _, _ = fit_all(p, th, sp, len(th), [truth])
    s0 = (p[0], th[0], np.clip(sp[0], 0.0, V_MAX), np.zeros(39), np.zeros(39), np.zeros(39))
    ps, ths = simulate(s0, fits[truth], truth, rng, BURN + SYNTH_STEPS, switching=(name == "T5"))
    ps, ths = ps[BURN:], ths[BURN:]
    v = np.zeros(ths.shape)
    v[1:] = np.hypot(*np.moveaxis(ps[1:] - ps[:-1], -1, 0)) / DT
    v[0] = v[1]
    out = analyse(ps, ths, v, 42_100 + int(name[1]))
    out.update({"name": name, "truth": truth})
    return out


def _real(args):
    n, rec, kind = args
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    out = analyse(p, th, np.where(sp > 100.0, 0.0, sp), 42_500 + 10 * n + rec + (5 if kind == "virtual" else 0),
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
    (out_dir / "e42_elimination_cohesion_gate_raw.json").write_text(json.dumps(rows))
    text = analyze(rows)
    from experiments.e42_figures import make_figures
    make_figures(rows, out_dir / "e42_figures", text)
    return text


def pairs(rows):
    real = {(r["N"], r["rec"]): r for r in rows["real"] if r["kind"] == "real"}
    virt = {(r["N"], r["rec"]): r for r in rows["real"] if r["kind"] == "virtual"}
    return [(real[k], virt[k]) for k in real]


def gate_open(v):
    return v["answer"] == ["none"]


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

    ratio = sum(r["cost_pruned"] for r in in_class) / sum(r["cost_abc"] for r in in_class)
    t5 = syn["T5"]
    g4 = sum(r["cohesion_coef"]["k6-AC"][0] > 0 for r in real)
    sp = [r["sim_stats"][r["shown"]][3] / r["obs_stats"][3] for r in real]
    g5 = sum(abs(x - 1) <= 0.25 for x in sp)
    g6 = sum(any(r["pass2"].values()) for r in real)
    g7 = sum(v["pass2"]["none"] for v in virt)
    g8 = sum(gate_open(v) for v in virt)
    return [
        ("G1", "truth survives stages 0-2 (T0-T4)", all(r["truth"] in r["surv2"] for r in in_class),
         ", ".join(f"{r['name']}:{'yes' if r['truth'] in r['surv2'] else 'no'}" for r in in_class)),
        ("G2", "answer <= 3, contains truth's rule (T0-T4)", all(answer_ok(r) for r in in_class),
         ", ".join(f"{r['name']}:{'+'.join(r['answer']) or '-'}" for r in in_class)),
        ("G3", "T5: micro-best fails stage 2", not t5["pass2"][t5["micro_best"]],
         f"{t5['micro_best']}, max q {max(t5['q2'][t5['micro_best']]):.2f}"),
        ("G4", "cohesion coefficient > 0, k6-AC (>= 7/8 real)", g4 >= 7, f"{g4}/8"),
        ("G5", "simulated spread within 25% (>= 6/8 real)", g5 >= 6,
         "sim/obs " + ", ".join(f"{x:.2f}" for x in sp)),
        ("G6", "a rule passes stage 2 (>= 4/8 real)", g6 >= 4, f"{g6}/8"),
        ("G7", "virtual: 'none' passes stage 2 (>= 7/8)", g7 >= 7, f"{g7}/8"),
        ("G8", "gate open: virtual answer is 'none' (>= 4/8)", g8 >= 4, f"{g8}/8"),
        ("G9", "pruned cost <= 0.5 x ABC-all (T0-T4)", ratio <= 0.5, f"ratio {ratio:.2f}"),
    ]


def analyze(rows) -> str:
    syn = rows["synthetic"]
    pr = pairs(rows)
    lines = ["Experiment 42: equivalence-margin stage 2, cohesion term, virtual-group gate", "",
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
    lines += ["", "  Cohesion coefficients, k6-AC (turn, speed):"]
    for r, _ in pr:
        lines.append(f"  {r['N']:3d}{r['rec']:4d}  " + "  ".join(f"{x:+.4f}" for x in r["cohesion_coef"]["k6-AC"]))
    claims = [(r, v) for r, v in pr if gate_open(v) and r["answer"]]
    lines += ["", "  Gated rule claims on real fish (exploratory): "
              + ("; ".join(f"{r['N']} fish #{r['rec']}: {', '.join(r['answer'])}" for r, _ in claims)
                 or "none")]
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(rows):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
