"""Experiment 39: multi-scale elimination of candidate interaction rules.

Question: can information from several scales rule out impossible
explanations faster than the conventional search, which simulates every
candidate model and compares it with the data?

Candidates (25). Each fish's 0.1 s heading change is
    own terms (bias, two wall terms, two lagged own turns; as in e32)
  + sum over neighbours j of the selected pairwise terms
      A  alignment   sin(theta_j - theta_i)
      C  attraction  sin(bearing_ij - theta_i)
      R  repulsion   sin(bearing_ij - theta_i) * exp(-d_ij / 2 cm)
  + residual noise (resampled from the fit).
Neighbour rules: topological k = 1, 3, 6 nearest; metric r = 5, 10, 20 cm.
Term sets: A, C, AC, ACR. Plus "none" (the twin). Every candidate is fitted
by least squares on the first 70 % (cheap; shared by every search method).

Family theorem (lean/InteractionShare.lean, eliminate_symmetric_alignment).
Metric alignment with an odd response is antisymmetric, so it never changes
the group's total turn: Omega = sum_i turn_i equals the twin's, sum_i own_i.
The family F_anti = {none, r5-A, r10-A, r20-A} is therefore eliminated at
once if the group-summed pairwise features predict the part of Omega that
the twin leaves (held-out, time steps where every fish moves).

Pipeline (pruned search), cheapest stage first:
  Stage 0  family test: z0 = mean(e_twin^2 - e_full^2) / block SE (20 blocks)
           on the last 30 %; F_anti eliminated if z0 > 3.
  Stage 1  short-horizon macro checks from observed states: 150 test states,
           4 replicas, horizons 1 s and 3 s, four macro variables
           (polarization, milling, mean distance to centroid, mean
           nearest-neighbour distance). Squared errors are scaled per
           (variable, horizon) by the median candidate's mean error and
           summed; a candidate is eliminated if its total exceeds the best
           survivor's by z > 3 (block SE, 15 blocks). Runs batched.
  Stage 2  long free simulation (2 runs x 1800 s, constant median speeds,
           60 s burn-in): mean and SD of polarization, mean milling, spread
           and nearest-neighbour distance vs the whole recording (10 blocks
           each); eliminated if any |z| > 4.
  Answer   the final survivors with the fewest terms.
Conventional search ("ABC-all"): Stage 2 on all 25 candidates.
Conventional micro selection: held-out one-step turn error; candidates
within z <= 3 of the best are "indistinguishable".
Cost: wall-clock seconds of the search (stages 0-2 on the candidates each
reaches, against stage 2 on all). Fitting and features are identical for
both searches; the ratio including them is reported too. Stage-1 size was
set from step timings measured before any result (39 fish: one simulated
step 134 us; a batched step over 2400 groups 0.19 s, which made the first
design, 300 states x 8 replicas, as costly as stage 2 itself).

Part A, synthetic schools with a known rule, 39 fish, 3600 s, coefficients
fitted to Puy recording N = 39 / 0, each fish at its median speed:
  T0 none, T1 r10-A (in F_anti), T2 k3-AC, T3 r10-AC, T4 k6-ACR,
  T5 out of class: k3-AC whose social terms are switched between 0 and 2x
     by a group-wide two-state Markov mode (mean dwell 60 s).
Part B, the 8 real Puy recordings (full pipeline + ABC-all) and their
virtual groups (stages 0-1 only).

Predictions, fixed before running:
  A1  No false elimination: the true rule survives stages 0-2 in T0-T4.
  A2  Stage 0 is right in T0-T4: F_anti kept for T0, T1; eliminated for
      T2, T3, T4.
  A3  Stages 0-1 eliminate >= 60 % of the 25 candidates, mean over T1-T4.
  A4  Precision: in T0-T4 the answer (simplest final survivors) contains the
      truth and has <= 3 candidates.
  A5  Speed: pruned search cost <= 0.5 x ABC-all search cost, summed over
      T0-T4.
  A6  Honest baseline: micro selection keeps the truth in its
      indistinguishable set in T0-T4 (expected PASS: with the right model
      class, one-step likelihood is efficient).
  A7  Out of class (T5): the micro-selected best candidate fails stage 2,
      i.e. the multi-scale checks flag a misspecification that micro
      selection cannot.
  B1  Real schools: stage 0 eliminates F_anti (including "none") in >= 6 of 8.
  B2  Virtual groups: stage 0 keeps F_anti and "none" survives stage 1 in
      all 8.
  B3  Real schools: stages 0-1 eliminate >= 50 % of candidates in >= 6 of 8.
  B4  Real schools: pruned search cost <= 0.5 x ABC-all search cost, summed
      over the 8.
Exploratory, no prediction: which rules survive on real fish, and whether any
candidate passes stage 2 there (constant-speed simulations may not
reproduce real spacing; that would be reported as "space eliminated").
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e31_real_fish import ARENA, MIN_SPEED, TRAIN, kinematics, virtual, wall_terms
from experiments.e32_fish_closed_twin import advance, turns
from experiments.fish_data import RECORDINGS, load

CENTRE = False
N_OWN = 5

RULES = ("k1", "k3", "k6", "r5", "r10", "r20")
TERMSETS = {"A": (0,), "C": (1,), "AC": (0, 1), "ACR": (0, 1, 2)}
CANDIDATES = ["none"] + [f"{r}-{t}" for r in RULES for t in TERMSETS]
ANTI = ("none", "r5-A", "r10-A", "r20-A")
TRUTHS = {"T0": "none", "T1": "r10-A", "T2": "k3-AC", "T3": "r10-AC", "T4": "k6-ACR",
          "T5": "k3-AC"}

N_STATES, REPLICAS, HORIZONS = 150, 4, (10, 30)
SIM_RUNS, SIM_STEPS, BURN = 2, 18000, 600
SYNTH_STEPS, SWITCH_P = 36000, 1 / 600
Z0, Z1, Z2 = 3.0, 3.0, 4.0


def own_features(p, th, l1, l2):
    """Bias, two wall terms, two lagged own turns; with CENTRE, a pull towards the tank centre."""
    cols = [np.ones(th.shape + (1,)), wall_terms(p, th), l1[..., None], l2[..., None]]
    if CENTRE:
        v = ARENA / 2 - p
        cols.append((np.sin(np.arctan2(v[..., 1], v[..., 0]) - th)
                     * np.hypot(v[..., 0], v[..., 1]) / (ARENA / 2))[..., None])
    return np.concatenate(cols, -1)


def nterms(h):
    return 0 if h == "none" else len(h.split("-")[1])


def block_se(d, nb):
    m = np.array([b.mean() for b in np.array_split(d, nb)])
    return m.std(ddof=1) / np.sqrt(nb)


def block_z(d, nb):
    se = block_se(d, nb)
    return 0.0 if se == 0 else float(d.mean() / se)


def pair_features(p, theta, rule, chunk=400):
    """Neighbour sums of alignment, attraction and repulsion terms: shape (..., N, 3)."""
    lead, n = theta.shape[:-1], theta.shape[-1]
    p2, th2 = p.reshape(-1, n, 2), theta.reshape(-1, n)
    out = np.empty((len(th2), n, 3))
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
        sb = np.sin(np.arctan2(diff[..., 1], diff[..., 0]) - th[:, :, None]) * w
        out[a:a + chunk, :, 0] = (np.sin(th[:, None, :] - th[:, :, None]) * w).sum(-1)
        out[a:a + chunk, :, 1] = sb.sum(-1)
        out[a:a + chunk, :, 2] = (sb * np.exp(-dist / 2.0)).sum(-1)
    return out.reshape(lead + (n, 3))


def macro(p, theta, chunk=2000):
    """Polarization, milling, mean distance to centroid, mean nearest-neighbour distance."""
    lead, n = theta.shape[:-1], theta.shape[-1]
    p2, th2 = p.reshape(-1, n, 2), theta.reshape(-1, n)
    out = np.empty((len(th2), 4))
    eye = np.eye(n, dtype=bool)
    for a in range(0, len(th2), chunk):
        pp, th = p2[a:a + chunk], th2[a:a + chunk]
        r = pp - pp.mean(1, keepdims=True)
        rn = np.hypot(r[..., 0], r[..., 1]) + 1e-9
        diff = pp[:, None, :, :] - pp[:, :, None, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        dist[:, eye] = np.inf
        out[a:a + chunk, 0] = np.abs(np.exp(1j * th).mean(-1))
        out[a:a + chunk, 1] = np.abs(((r[..., 0] * np.sin(th) - r[..., 1] * np.cos(th)) / rn).mean(-1))
        out[a:a + chunk, 2] = rn.mean(-1)
        out[a:a + chunk, 3] = dist.min(-1).mean(-1)
    return out.reshape(lead + (4,))


def design(own, feats, h):
    if h == "none":
        return own
    rule, ts = h.split("-")
    return np.concatenate([own, feats[rule][..., TERMSETS[ts]]], -1)


def model_step(state, coef, h, res, rng, gain=1.0):
    p, th, sp, l1, l2 = state
    turn = own_features(p, th, l1, l2) @ coef[:N_OWN] + rng.choice(res, th.shape)
    if h != "none" and gain != 0.0:
        rule, ts = h.split("-")
        turn = turn + gain * (pair_features(p, th, rule)[..., TERMSETS[ts]] @ coef[N_OWN:])
    p, th = advance(p, th, sp, turn)
    return p, th, sp, turn, l1


def simulate(state, coef, h, res, rng, steps, record_every=1, switching=False):
    mode, ps, ths = 1, [], []
    for t in range(steps):
        if switching and rng.random() < SWITCH_P:
            mode = 1 - mode
        if t % record_every == 0:
            ps.append(state[0]), ths.append(state[1])
        state = model_step(state, coef, h, res, rng, gain=2.0 * mode if switching else 1.0)
    return np.array(ps), np.array(ths)


def stage2_stats(series, nb):
    return np.array([[b[:, 0].mean(), b[:, 0].std(), b[:, 1].mean(), b[:, 2].mean(), b[:, 3].mean()]
                     for b in np.array_split(series, nb)])


def analyse(p, th, sp, seed, full=True):
    """Every search method on one recording (real, virtual or synthetic)."""
    rng = np.random.default_rng(seed)
    T, n = th.shape
    cut = int(TRAIN * T)
    t_start = time.perf_counter()
    l1, l2, ok_in = turns(th, sp)
    own = own_features(p, th, l1, l2)
    feats = {r: pair_features(p, th, r) for r in RULES}
    y, okf = l1[1:], ok_in[1:]
    train = okf.copy()
    train[cut:] = False
    fits = {}
    for h in CANDIDATES:
        x = design(own, feats, h)[:-1]
        coef = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
        fits[h] = (coef, y[train] - x[train] @ coef)
    shared = time.perf_counter() - t_start

    # Stage 0: does the pairwise structure steer the group's total turn?
    t0 = time.perf_counter()
    union = np.concatenate([own] + [feats[r] for r in RULES], -1)[:-1]
    cu = np.linalg.lstsq(union[train], y[train], rcond=None)[0]
    allv = okf.all(1)
    resid = (y - own[:-1] @ cu[:N_OWN])[allv].sum(1)
    fsum = union[allv][..., N_OWN:].sum(1)
    tr = np.nonzero(allv)[0] < cut
    xg = np.column_stack([np.ones(len(resid)), fsum])
    bg = np.linalg.lstsq(xg[tr], resid[tr], rcond=None)[0]
    e_full = resid[~tr] - xg[~tr] @ bg
    e_twin = resid[~tr] - resid[tr].mean()
    z0 = block_z(e_twin ** 2 - e_full ** 2, 20)
    s_omega = float(1 - np.mean(e_full ** 2) / np.mean(e_twin ** 2))
    surv0 = [h for h in CANDIDATES if not (z0 > Z0 and h in ANTI)]
    time0 = time.perf_counter() - t0
    del union

    # Stage 1: short-horizon macro checks from observed states, batched.
    idx = np.linspace(cut, T - max(HORIZONS) - 1, N_STATES).astype(int)
    obs = np.concatenate([macro(p[idx + s], th[idx + s]) for s in HORIZONS], -1)
    err, time1 = {}, {}
    for k, h in enumerate(CANDIDATES):
        t1 = time.perf_counter()
        coef, res = fits[h]
        r1 = np.random.default_rng(seed * 100 + k)
        state = tuple(np.repeat(a[idx][:, None], REPLICAS, 1) for a in (p, th, sp, l1, l2))
        pred = []
        for s in range(1, max(HORIZONS) + 1):
            state = model_step(state, coef, h, res, r1)
            if s in HORIZONS:
                pred.append(macro(state[0], state[1]).mean(1))
        err[h] = (obs - np.concatenate(pred, -1)) ** 2
        time1[h] = time.perf_counter() - t1

    def stage1(pool):
        scale = np.median([err[h].mean(0) for h in pool], axis=0)
        tot = {h: (err[h] / scale).sum(1) for h in pool}
        best = min(pool, key=lambda h: tot[h].mean())
        z = {h: block_z(tot[h] - tot[best], 15) for h in pool}
        return [h for h in pool if z[h] <= Z1], z

    surv1, z1 = stage1(surv0)

    # Conventional micro selection: held-out one-step turn error.
    test = okf.copy()
    test[:cut] = False
    step_err = {}
    for h in CANDIDATES:
        x = design(own, feats, h)[:-1]
        step_err[h] = np.where(test, (y - x @ fits[h][0]) ** 2, 0.0)[cut:].sum(1)
    micro_best = min(CANDIDATES, key=lambda h: step_err[h].mean())
    z_micro = {h: block_z(step_err[h] - step_err[micro_best], 20) for h in CANDIDATES}
    del feats, own

    out = {"T": T, "n": n, "z0": z0, "S_omega": s_omega, "surv0": surv0, "surv1": surv1,
           "z1": z1, "micro_best": micro_best, "z_micro": z_micro, "shared_s": shared,
           "stage0_s": time0, "stage1_s": time1}
    if not full:
        return out

    # Stage 2 (and ABC-all): long free simulations of every candidate.
    obs_series = macro(p[::10], th[::10])
    o = stage2_stats(obs_series, 10)
    o_mean, o_se = o.mean(0), o.std(0, ddof=1) / np.sqrt(10)
    speed = np.median(sp, axis=0)
    starts = (0, cut)
    z2, time2, sim_means = {}, {}, {}
    for k, h in enumerate(CANDIDATES):
        t2 = time.perf_counter()
        coef, res = fits[h]
        r2 = np.random.default_rng(seed * 1000 + k)
        blocks = []
        for run in range(SIM_RUNS):
            s0 = (p[starts[run]], th[starts[run]], speed, np.zeros(n), np.zeros(n))
            ps, ths = simulate(s0, coef, h, res, r2, BURN + SIM_STEPS, record_every=10)
            blocks.append(stage2_stats(macro(ps[BURN // 10:], ths[BURN // 10:]), 10 // SIM_RUNS))
        sb = np.concatenate(blocks)
        s_mean, s_se = sb.mean(0), sb.std(0, ddof=1) / np.sqrt(len(sb))
        z2[h] = (np.abs(s_mean - o_mean) / np.sqrt(o_se ** 2 + s_se ** 2)).tolist()
        sim_means[h] = s_mean.tolist()
        time2[h] = time.perf_counter() - t2
    pass2 = {h: max(z2[h]) <= Z2 for h in CANDIDATES}
    surv2 = [h for h in surv1 if pass2[h]]
    fewest = min((nterms(h) for h in surv2), default=None)
    out.update({"z2": z2, "pass2": pass2, "surv2": surv2, "obs_stats": o_mean.tolist(),
                "sim_stats": sim_means, "stage2_s": time2,
                "answer": [h for h in surv2 if nterms(h) == fewest],
                "abc_survivors": [h for h in CANDIDATES if pass2[h]],
                "cost_pruned": time0 + sum(time1[h] for h in surv0) + sum(time2[h] for h in surv1),
                "cost_abc": sum(time2.values())})
    return out


def _synthetic(name):
    truth = TRUTHS[name]
    rng = np.random.default_rng(39_000 + int(name[1]))
    p, th, sp = kinematics(load(39, 0))
    sp = np.where(sp > 100.0, 0.0, sp)
    l1, l2, ok_in = turns(th, sp)
    own = own_features(p, th, l1, l2)
    feats = {r: pair_features(p, th, r) for r in RULES} if truth != "none" else {}
    x = design(own, feats, truth)[:-1]
    ok = ok_in[1:]
    coef = np.linalg.lstsq(x[ok], l1[1:][ok], rcond=None)[0]
    res = l1[1:][ok] - x[ok] @ coef
    del feats, own, x
    speed = np.maximum(np.median(sp, axis=0), MIN_SPEED)
    s0 = (p[0], th[0], speed, np.zeros(39), np.zeros(39))
    ps, ths = simulate(s0, coef, truth, res, rng, BURN + SYNTH_STEPS, switching=(name == "T5"))
    ps, ths = ps[BURN:], ths[BURN:]
    out = analyse(ps, ths, np.broadcast_to(speed, ths.shape).copy(), 39_100 + int(name[1]))
    out.update({"name": name, "truth": truth, "truth_coef": coef.tolist()})
    return out


def _real(args):
    n, rec, kind = args
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    out = analyse(p, th, np.where(sp > 100.0, 0.0, sp), 39_500 + 10 * n + rec,
                  full=(kind == "real"))
    out.update({"N": n, "rec": rec, "kind": kind})
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    global N_STATES, SIM_STEPS, SYNTH_STEPS, BURN
    out_dir = out_dir or Path("results")
    if quick:
        N_STATES, SIM_STEPS, SYNTH_STEPS, BURN = 60, 1200, 3000, 100
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    tasks = [(n, r, k) for n, recs in RECORDINGS.items() for r in recs for k in ("real", "virtual")]
    print(f"  {len(TRUTHS)} synthetic truths + {len(tasks)} real/virtual recordings ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 2))) as pool:
        synth = pool.map(_synthetic, TRUTHS)
        real = pool.map(_real, tasks)
        rows = {"synthetic": list(synth), "real": list(real)}
    (out_dir / "e39_multiscale_elimination_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows)


def analyze(rows) -> str:
    syn = {r["name"]: r for r in rows["synthetic"]}
    real = [r for r in rows["real"] if r["kind"] == "real"]
    virt = [r for r in rows["real"] if r["kind"] == "virtual"]
    nc = len(CANDIDATES)
    lines = ["Experiment 39: multi-scale elimination of candidate interaction rules", "",
             "  Part A, synthetic schools with a known rule (39 fish):",
             f"  {'':4}{'truth':>8}{'z0':>8}{'S_Omega':>9}{'left 0':>8}{'left 1':>8}{'left 2':>8}"
             f"{'ABC left':>10}{'cost ratio':>12}  answer | micro best"]
    for name, r in syn.items():
        lines.append(f"  {name:4}{r['truth']:>8}{r['z0']:8.1f}{r['S_omega']:9.4f}{len(r['surv0']):8d}"
                     f"{len(r['surv1']):8d}{len(r['surv2']):8d}{len(r['abc_survivors']):10d}"
                     f"{r['cost_pruned'] / r['cost_abc']:12.2f}  {', '.join(r['answer']) or '(none)'}"
                     f" | {r['micro_best']}")
    lines += ["", "  Part B, real schools (Puy et al. 2024):",
              f"  {'N':>3}{'rec':>4}{'z0':>8}{'S_Omega':>9}{'left 0':>8}{'left 1':>8}{'left 2':>8}"
              f"{'ABC left':>10}{'cost ratio':>12}  stage-1 survivors | answer | micro best"]
    for r in real:
        lines.append(f"  {r['N']:3d}{r['rec']:4d}{r['z0']:8.1f}{r['S_omega']:9.4f}{len(r['surv0']):8d}"
                     f"{len(r['surv1']):8d}{len(r['surv2']):8d}{len(r['abc_survivors']):10d}"
                     f"{r['cost_pruned'] / r['cost_abc']:12.2f}  {', '.join(r['surv1'])} | "
                     f"{', '.join(r['answer']) or '(space eliminated)'} | {r['micro_best']}")
    lines += ["", "  Virtual groups (stages 0-1):",
              f"  {'N':>3}{'rec':>4}{'z0':>8}{'left 1':>8}  'none' survives"]
    for r in virt:
        lines.append(f"  {r['N']:3d}{r['rec']:4d}{r['z0']:8.1f}{len(r['surv1']):8d}  {'none' in r['surv1']}")

    in_class = [syn[t] for t in ("T0", "T1", "T2", "T3", "T4")]
    a1 = all(r["truth"] in r["surv2"] for r in in_class)
    a2 = all((r["z0"] > Z0) == (r["truth"] not in ANTI) for r in in_class)
    elim = [1 - len(syn[t]["surv1"]) / nc for t in ("T1", "T2", "T3", "T4")]
    a3 = np.mean(elim) >= 0.6
    a4 = all(r["truth"] in r["answer"] and len(r["answer"]) <= 3 for r in in_class)
    incl = lambda rs: (sum(r["shared_s"] + r["cost_pruned"] for r in rs)
                       / sum(r["shared_s"] + r["cost_abc"] for r in rs))
    ratio_a = sum(r["cost_pruned"] for r in in_class) / sum(r["cost_abc"] for r in in_class)
    a5 = ratio_a <= 0.5
    a6 = all(r["z_micro"][r["truth"]] <= 3 for r in in_class)
    t5 = syn["T5"]
    a7 = not t5["pass2"][t5["micro_best"]]
    b1n = sum(r["z0"] > Z0 for r in real)
    b2 = all(r["z0"] <= Z0 and "none" in r["surv1"] for r in virt)
    b3n = sum(1 - len(r["surv1"]) / nc >= 0.5 for r in real)
    ratio_b = sum(r["cost_pruned"] for r in real) / sum(r["cost_abc"] for r in real)
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(a1)}  A1 truth survives stages 0-2 in T0-T4",
              f"  {tag(a2)}  A2 stage 0 right in T0-T4 [z0: "
              + ", ".join(f"{r['z0']:.1f}" for r in in_class) + "]",
              f"  {tag(a3)}  A3 stages 0-1 eliminate >= 60 % (T1-T4) [{np.mean(elim):.0%}]",
              f"  {tag(a4)}  A4 answer contains truth, <= 3 candidates [sizes "
              + ", ".join(str(len(r["answer"])) for r in in_class) + "]",
              f"  {tag(a5)}  A5 pruned search cost <= 0.5 x ABC-all (T0-T4) [{ratio_a:.2f}; "
              f"including fitting {incl(in_class):.2f}]",
              f"  {tag(a6)}  A6 micro selection keeps the truth (T0-T4) [z "
              + ", ".join(f"{r['z_micro'][r['truth']]:.1f}" for r in in_class) + "]",
              f"  {tag(a7)}  A7 T5: micro-best ({t5['micro_best']}) fails stage 2 "
              f"[max z {max(t5['z2'][t5['micro_best']]):.1f}]",
              f"  {tag(b1n >= 6)}  B1 real: stage 0 eliminates F_anti in >= 6 of 8 [{b1n}/8]",
              f"  {tag(b2)}  B2 virtual: F_anti kept and 'none' survives stage 1 in all 8",
              f"  {tag(b3n >= 6)}  B3 real: stages 0-1 eliminate >= 50 % in >= 6 of 8 [{b3n}/8]",
              f"  {tag(ratio_b <= 0.5)}  B4 real: pruned search cost <= 0.5 x ABC-all "
              f"[{ratio_b:.2f}; including fitting {incl(real):.2f}]"]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
