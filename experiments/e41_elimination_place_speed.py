"""Experiment 41: multi-scale elimination with place preferences, speed dynamics
and an adaptive stage 1.

e39/e40 left three problems:
  1. False alarms on virtual 8-fish groups (no interaction), not removed by
     a tank-centre pull (e40). Diagnosis: richer spatial structure in each
     fish's own behaviour, which pairwise terms can mimic.
  2. Stage 2 eliminated every rule on real fish: constant-speed simulations
     spread 1.3-2.3x wider than real schools (no cohesion).
  3. The pruned search was only cheaper when there was something to prune
     (cost ratio 0.72 and 0.65 against a predicted <= 0.5).

Changes (everything else as in e39: 25 candidates, stage 0 family test,
macro variables, horizons, thresholds 3 / 4, seeds scheme, synthetic truths
T0-T5 refitted on recording 39/0 with the new own model):
  Own turning  + place-preference field: (1, x, y, x^2, y^2, xy) x (sin theta,
               cos theta), x and y scaled to [-1, 1]; this contains any
               quadratic preferred-direction field, including a centre pull.
  Speed        every fish's speed now evolves: dv = s0 + s1 v + s2 dv_prev +
               noise (resampled), clipped to [0, 60] cm/s. Candidates with
               attraction (C) also get a speed response to neighbours ahead,
               sum_j w_ij cos(bearing_ij - theta_i) (speed up when they lead).
  Stage 1      adaptive: the 150 test states are split into 5 interleaved
               batches of 30; after each batch from the second on, a
               candidate is dropped if z > 4 (early) or, after the last
               batch, z > 3.
  Answer       nested-minimal survivors: a survivor is dropped if a simpler
               nested candidate (same neighbour rule, subset of terms, or
               "none") also survived. (e40's fewest-terms rule compared across
               neighbour rules and dropped a true weak term.)

Predictions, fixed before running:
  F1  The true rule survives stages 0-2 in T0-T4.
  F2  The answer has <= 3 candidates and contains the truth's neighbour rule
      with a subset of its terms (or "none" for T0), in T0-T4.
  F3  Stage 0 is right in T0-T4 (F_anti kept for T0, T1; eliminated for T2-T4).
  F4  Pruned search cost <= 0.5 x ABC-all search cost, summed over T0-T4.
  F5  T5 (out of class): the micro-selected best candidate fails stage 2.
  F6  Virtual groups: F_anti kept and "none" survives stage 1 in >= 7 of 8.
  F7  Real schools: stage 0 fires in >= 6 of 8 real and <= 1 of 8 virtual.
  F8  Cohesion: at least one candidate passes stage 2 in >= 4 of 8 real
      recordings.
Exploratory: which rules survive on real fish.

Figures (results/e41_figures/) are drawn from the raw results after the run.
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
from experiments.e39_multiscale_elimination import (ANTI, CANDIDATES, RULES, SWITCH_P, TERMSETS,
                                                    TRUTHS, block_z, macro, nterms, stage2_stats)
from experiments.fish_data import RECORDINGS, load

DT = 0.1
N_OWN, N_OWN_V, V_MAX = 17, 3, 60.0
N_STATES, REPLICAS, HORIZONS, BATCHES = 150, 4, (10, 30), 5
SIM_RUNS, SIM_STEPS, BURN = 2, 18000, 600
SYNTH_STEPS = 36000
Z0, Z1, Z1_EARLY, Z2 = 3.0, 3.0, 4.0, 4.0
SNAP = 30


def features(p, theta, rule, chunk=400):
    """Neighbour sums: alignment, attraction, repulsion, and 'ahead' (cos bearing) for speed."""
    lead, n = theta.shape[:-1], theta.shape[-1]
    p2, th2 = p.reshape(-1, n, 2), theta.reshape(-1, n)
    out = np.empty((len(th2), n, 4))
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
    return out.reshape(lead + (n, 4))


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
    return np.concatenate([own_v, feats[h.split("-")[0]][..., 3:4]], -1)


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
            dvn = dvn + gain * cv[N_OWN_V] * f[..., 3]
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


def analyse(p, th, sp, seed, full=True):
    T, n = th.shape
    cut = int(TRAIN * T)
    t_start = time.perf_counter()
    fits, own, feats, (l1, l2, ok_in, dvp) = fit_all(p, th, sp, cut, CANDIDATES)
    y, okf = l1[1:], ok_in[1:]
    train = okf.copy()
    train[cut:] = False
    shared = time.perf_counter() - t_start

    t0 = time.perf_counter()
    union = np.concatenate([own] + [feats[r][..., :3] for r in RULES], -1)[:-1]
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

    # Stage 1, adaptive.
    idx = np.linspace(cut, T - max(HORIZONS) - 1, N_STATES).astype(int)
    obs = np.concatenate([macro(p[idx + s], th[idx + s]) for s in HORIZONS], -1)
    err = {h: np.full((N_STATES, 2 * 4), np.nan) for h in surv0}
    time1 = {h: 0.0 for h in surv0}
    dropped_at, z1 = {}, {}
    alive, seen = list(surv0), np.zeros(N_STATES, bool)
    for b in range(BATCHES):
        bi = np.arange(b, N_STATES, BATCHES)
        for k, h in enumerate(alive):
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
    z_micro = {h: block_z(step_err[h] - step_err[micro_best], 20) for h in CANDIDATES}
    del feats, own

    out = {"T": T, "n": n, "z0": z0, "S_omega": s_omega, "surv0": surv0, "surv1": surv1,
           "z1": z1, "dropped_at": dropped_at, "micro_best": micro_best, "z_micro": z_micro,
           "shared_s": shared, "stage0_s": time0, "stage1_s": time1,
           "speed_coef": {h: fits[h][2].tolist() for h in ("none", "k3-AC", "k6-AC")}}
    if not full:
        return out

    obs_series = macro(p[::10], th[::10])
    o = stage2_stats(obs_series, 10)
    o_mean, o_se = o.mean(0), o.std(0, ddof=1) / np.sqrt(10)
    starts = (0, cut)
    show = best1 if surv1 else micro_best
    z2, time2, sim_means, snap = {}, {}, {}, {}
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
            if h == show and run == 0:
                snap = {"sim_p": ps[-SNAP:].tolist(), "sim_th": ths[-SNAP:].tolist(),
                        "sim_series": ser[::2].tolist()}
        sb = np.concatenate(blocks)
        s_mean, s_se = sb.mean(0), sb.std(0, ddof=1) / np.sqrt(len(sb))
        z2[h] = (np.abs(s_mean - o_mean) / np.sqrt(o_se ** 2 + s_se ** 2)).tolist()
        sim_means[h] = s_mean.tolist()
        time2[h] = time.perf_counter() - t2
    pass2 = {h: max(z2[h]) <= Z2 for h in CANDIDATES}
    surv2 = [h for h in surv1 if pass2[h]]
    obs_frames = np.arange(T - 10 * SNAP, T, 10)
    out.update({"z2": z2, "pass2": pass2, "surv2": surv2, "obs_stats": o_mean.tolist(),
                "obs_se": o_se.tolist(), "sim_stats": sim_means, "stage2_s": time2,
                "answer": nested_min(surv2), "abc_survivors": [h for h in CANDIDATES if pass2[h]],
                "shown": show, **snap, "obs_p": p[obs_frames].tolist(),
                "obs_th": th[obs_frames].tolist(), "obs_series": obs_series[::2].tolist(),
                "cost_pruned": time0 + sum(time1.values()) + sum(time2[h] for h in surv1),
                "cost_abc": sum(time2.values())})
    return out


def _synthetic(name):
    truth = TRUTHS[name]
    rng = np.random.default_rng(41_000 + int(name[1]))
    p, th, sp = kinematics(load(39, 0))
    sp = np.where(sp > 100.0, 0.0, sp)
    fits, _, _, _ = fit_all(p, th, sp, len(th), [truth])
    s0 = (p[0], th[0], np.clip(sp[0], 0.0, V_MAX), np.zeros(39), np.zeros(39), np.zeros(39))
    ps, ths = simulate(s0, fits[truth], truth, rng, BURN + SYNTH_STEPS, switching=(name == "T5"))
    ps, ths = ps[BURN:], ths[BURN:]
    v = np.zeros(ths.shape)
    v[1:] = np.hypot(*np.moveaxis(ps[1:] - ps[:-1], -1, 0)) / DT
    v[0] = v[1]
    out = analyse(ps, ths, v, 41_100 + int(name[1]))
    out.update({"name": name, "truth": truth})
    return out


def _real(args):
    n, rec, kind = args
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    out = analyse(p, th, np.where(sp > 100.0, 0.0, sp), 41_500 + 10 * n + rec, full=(kind == "real"))
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
    (out_dir / "e41_elimination_place_speed_raw.json").write_text(json.dumps(rows))
    text = analyze(rows)
    from experiments.e41_figures import make_figures
    make_figures(rows, out_dir / "e41_figures", text)
    return text


def predictions(rows):
    syn = {r["name"]: r for r in rows["synthetic"]}
    real = [r for r in rows["real"] if r["kind"] == "real"]
    virt = [r for r in rows["real"] if r["kind"] == "virtual"]
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
    f6n = sum(r["z0"] <= Z0 and "none" in r["surv1"] for r in virt)
    f7r, f7v = sum(r["z0"] > Z0 for r in real), sum(r["z0"] > Z0 for r in virt)
    f8n = sum(any(r["pass2"].values()) for r in real)
    return [
        ("F1", "truth survives stages 0-2 (T0-T4)", all(r["truth"] in r["surv2"] for r in in_class),
         ", ".join(f"{r['name']}:{'yes' if r['truth'] in r['surv2'] else 'no'}" for r in in_class)),
        ("F2", "answer <= 3, contains truth's rule (T0-T4)", all(answer_ok(r) for r in in_class),
         ", ".join(f"{r['name']}:{'+'.join(r['answer']) or '-'}" for r in in_class)),
        ("F3", "stage 0 right (T0-T4)", all((r["z0"] > Z0) == (r["truth"] not in ANTI) for r in in_class),
         "z0 " + ", ".join(f"{r['z0']:.1f}" for r in in_class)),
        ("F4", "pruned cost <= 0.5 x ABC-all (T0-T4)", ratio <= 0.5, f"ratio {ratio:.2f}"),
        ("F5", "T5: micro-best fails stage 2", not t5["pass2"][t5["micro_best"]],
         f"{t5['micro_best']}, max z {max(t5['z2'][t5['micro_best']]):.1f}"),
        ("F6", "virtual: F_anti kept, 'none' survives (>= 7/8)", f6n >= 7, f"{f6n}/8"),
        ("F7", "stage 0 fires: real >= 6/8, virtual <= 1/8", f7r >= 6 and f7v <= 1,
         f"real {f7r}/8, virtual {f7v}/8"),
        ("F8", "cohesion: a rule passes stage 2 (>= 4/8 real)", f8n >= 4, f"{f8n}/8"),
    ]


def analyze(rows) -> str:
    syn = rows["synthetic"]
    real = [r for r in rows["real"] if r["kind"] == "real"]
    virt = [r for r in rows["real"] if r["kind"] == "virtual"]
    lines = ["Experiment 41: elimination with place preferences, speed dynamics, adaptive stage 1", "",
             "  Part A, synthetic schools with a known rule (39 fish):",
             f"  {'':4}{'truth':>8}{'z0':>7}{'left 0':>8}{'left 1':>8}{'left 2':>8}{'ABC':>5}"
             f"{'cost':>7}  answer | micro best"]
    for r in syn:
        lines.append(f"  {r['name']:4}{r['truth']:>8}{r['z0']:7.1f}{len(r['surv0']):8d}{len(r['surv1']):8d}"
                     f"{len(r['surv2']):8d}{len(r['abc_survivors']):5d}"
                     f"{r['cost_pruned'] / r['cost_abc']:7.2f}  {', '.join(r['answer']) or '(none)'}"
                     f" | {r['micro_best']}")
    lines += ["", "  Part B, real schools (Puy et al. 2024):",
              f"  {'N':>3}{'rec':>4}{'z0':>7}{'left 0':>8}{'left 1':>8}{'left 2':>8}{'ABC':>5}{'cost':>7}"
              "  stage-1 survivors | answer | micro best"]
    for r in real:
        lines.append(f"  {r['N']:3d}{r['rec']:4d}{r['z0']:7.1f}{len(r['surv0']):8d}{len(r['surv1']):8d}"
                     f"{len(r['surv2']):8d}{len(r['abc_survivors']):5d}"
                     f"{r['cost_pruned'] / r['cost_abc']:7.2f}  {', '.join(r['surv1'])} | "
                     f"{', '.join(r['answer']) or '(space eliminated)'} | {r['micro_best']}")
    lines += ["", "  Real schools, observed vs shown candidate (polarization mean, SD, milling, "
              "spread cm, NND cm):"]
    for r in real:
        lines.append(f"  {r['N']:3d}{r['rec']:4d}  obs " + " ".join(f"{x:6.2f}" for x in r["obs_stats"])
                     + f"   {r['shown']:>7} " + " ".join(f"{x:6.2f}" for x in r["sim_stats"][r["shown"]]))
    lines += ["", "  Virtual groups (stages 0-1):", f"  {'N':>3}{'rec':>4}{'z0':>7}{'left 1':>8}  'none' survives"]
    for r in virt:
        lines.append(f"  {r['N']:3d}{r['rec']:4d}{r['z0']:7.1f}{len(r['surv1']):8d}  {'none' in r['surv1']}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(rows):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
