"""Experiment 77: why the quadratic memory gap runs above the fitted fish twins.

e72's quadratic memory gap ranked real schools right but ran above the published shares of the fitted fish twins
(e33 adult fish, e38 zebrafish; median excess 0.14, up to 0.41). The roadmap read this as "what fuller twins
explain" and asked for a correction. Development on simulated schools only (seed 900; e32's fish model fitted to
adult recording 0, run with its own social terms, with other interaction rules, or as loners) did not support that
reading:
  * the memory twin was never above the true twin (at most +0.03), and far below it in a locked school (39 fish
    aligning to their 7 nearest neighbours' headings 0.5 s earlier: 0.01 against 0.995): there each fish's own past
    already carries the school's heading;
  * a linear twin from each fish's own walls and turns scored *lower* than the memory twin (0.66-0.88 against
    0.72-0.92): in a school, a fish's own position reveals the school's, so it absorbs interaction. Dropped;
  * the fitted full twin did not absorb interaction under misspecified rules (within 0.03 of the true twin except
    the locked school);
  * on these stationary schools, interleaved and chronological (first 70% / last 30%) splits agree within 0.016, and
    the full twin scored on phi^2 with e72's law reproduces e33's S_proj (on phi) within 0.05.
So nothing in simulation puts the memory gap above the fitted twin. The real excess must come from what simulation
lacks. Two candidates, tested here head to head on the real schools:
  split  e33/e38 trained on the first 70% of each recording and tested on the last 30%; e72 cross-fits over
         interleaved 60 s blocks. If schools drift within a recording, the chronological split lowers the share.
  twin   real fish reveal more to the full twin (walls, persistent turning) than to their heading memory.

Every share is on e72's target (polarization squared), law (cubic in the school's windowed polarization) and exact
independent-fish formula phi0^2 = (N + |sum m_i|^2 - sum |m_i|^2) / N^2, with member forecasts m_i from:
  memory  e72's memory twin (own mean heading + own memory weights; nothing fitted);
  full    e32's (square tank) or e38's (round tank) fish model: next turn on own walls, own two previous turns and
          4-nearest-neighbour social terms, fitted on the training part only (e76: nothing fitted sees the test
          part); the twin drops the social terms and is simulated 1 s ahead, 32 replicas, m_i = mean heading;
  true    (simulations) the generating model's own part.
and two splits: interleaved (3 folds of 60 s blocks, as e72) and chronological (first 70% / last 30%, as e33/e38).
Simulations: fresh seeds 910, 911; N 8 and 39; loners, the model's own interaction, metric alignment (k 0.1, 0.3),
delayed topological alignment (k 0.1, 0.3); 60 min each. Real: e72's 17 schools and 17 virtual groups.

Predictions, fixed before running:
  A1  Simulated loners: every share within +-0.03.
  A2  Stationary schools: chronological and interleaved splits agree within 0.03 (full twin), every school.
  A3  Bridge: true twin on phi^2 (chronological) within 0.05 of e33's S_proj (true twin) in >= 80% of schools.
  A4  The memory gap is not above the true twin: S_memory <= S_true + 0.05 in every school (interleaved).
  A5  The fitted full twin is within 0.03 of the true twin in >= 80% of schools (interleaved).
  B1  Real bridge: full twin on phi^2 (chronological) within 0.05 of the published S_proj (median over 17 schools).
  B2  Real virtual groups: |S_full| <= 0.1 in >= 80% (interleaved).
  B3  Drift: median (S_full interleaved - S_full chronological) >= 0.05 over real schools.
  B4  Twin content is small: median (S_memory - S_full), interleaved, <= 0.05 over real schools.
  B5  The correction: the memory gap scored on the chronological split is within 0.07 of the published S_proj
      (median |diff|; e72's interleaved value was 0.140).
B3 and B4 are the two explanations; B5 holds if the split is the cause.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments import e32_fish_closed_twin as e32
from experiments import e33_fish_projected_twin as e33
from experiments import e38_zebrafish as e38
from experiments import e72_polarization_gap as e72
from experiments.e31_real_fish import ARENA, TRAIN, kinematics, social_terms, virtual
from experiments.e31_real_fish import wall_terms as wall_square
from experiments.e67_memory_gap import _toeplitz, design
from experiments.e68_fish_memory_gap import WINDOWS, memories

SEED = 77
LEAD = 10
REPS = 32
OFFS, F, Y = design(WINDOWS, [0])
SIM_SEEDS = (910, 911)
RULES = (("model", 0.0), ("metric", 0.1), ("metric", 0.3), ("delayed", 0.1), ("delayed", 0.3))
SIM_STEPS = 36000


def unit(theta):
    return np.stack([np.cos(theta), np.sin(theta)], -1)


# ---------------------------------------------------------------- splits and the share

def layout(t):
    """Target rows (e72) and the two splits: interleaved 3-fold blocks, and one chronological 70/30 split."""
    rows = np.arange(-int(OFFS.min()), t - int(OFFS.max()))
    idx = np.arange(len(rows))
    fold = (idx // e72.BLOCK) % e72.FOLDS
    inter = [(fold != k, fold == k) for k in range(e72.FOLDS)]
    cut = idx < int(TRAIN * len(rows))
    return rows, {"interleaved": inter, "chrono": [(cut, ~cut)]}


def share(theta, rows, split, forecasts):
    """S on phi^2 with e72's law; forecasts[k] (rows, n, 2) are the twin's member forecasts for split k."""
    n = theta.shape[1]
    u = unit(theta)
    phi = np.sqrt((u.sum(1) ** 2).sum(-1)) / n
    target = phi[rows] ** 2
    pw = np.stack([sum(phi[rows + o] for o in OFFS[F[w] > 0]) / F[w].sum() for w in range(len(F))], 1)
    feats = np.column_stack([np.ones(len(rows))] + [pw ** p for p in (1, 2, 3)])
    eg, e0 = [], []
    for (tr, te), m in zip(split, forecasts):
        phi0 = (n + (m.sum(1) ** 2).sum(-1) - (m ** 2).sum((1, 2))) / n ** 2
        eg.append(target[te] - e72._ols(feats[tr], target[tr], feats[te]))
        e0.append(target[te] - e72._ols(feats[tr], phi0[tr], feats[te]))
    return float(1 - (np.concatenate(eg) ** 2).sum() / (np.concatenate(e0) ** 2).sum())


# ---------------------------------------------------------------- twins

def memory_twin(theta, rows):
    """e72's member forecasts: own mean heading plus own memory weights on own past (nothing fitted)."""
    u = unit(theta)
    mu = u.mean(0)
    o, _ = memories(theta)
    mo = _toeplitz(o, OFFS - OFFS.min())
    beta = np.linalg.solve(F @ mo @ F.T, F @ mo @ Y)
    sums = np.stack([sum((u - mu)[rows + o] for o in OFFS[F[w] > 0]) for w in range(len(F))], 1)
    return mu[None] + np.einsum("w,rwnd->rnd", beta, sums)


class Tank:
    """The fitted fish model of e32 (square tank) or e38 (round tank): own terms + social terms, and its twin."""

    def __init__(self, kind, radius=None, dt=None):
        self.kind, self.radius, self.dt = kind, radius, dt

    def design(self, p, theta, speed):
        if self.kind == "square":
            l1, l2, ok = e32.turns(theta, speed)
            own = np.concatenate([np.ones(theta.shape + (1,)), wall_square(p, theta), l1[..., None], l2[..., None]], -1)
            return np.concatenate([own, social_terms(p, theta)], -1), l1, l2, ok
        x, l1, ok = e38.features(p, theta, speed, self.radius)
        l2 = np.zeros_like(l1)
        l2[1:] = l1[:-1]
        return x, l1, l2, ok

    def fit(self, x, l1, ok, train_t):
        """Least squares of the next turn on the state, on time steps whose next step is in train_t."""
        y = l1[1:]
        use = ok[1:] & train_t[1:, None]
        coef = np.linalg.lstsq(x[:-1][use], y[use], rcond=None)[0]
        return coef[:e32.N_OWN], (y - x[:-1] @ coef)[use]

    def step(self, state, c, res, rng):
        if self.kind == "square":
            return e32.twin_step(state, c, res, rng)
        e38.DT_HOLDER[0] = self.dt
        return e38.step(state, c, res, rng, self.radius)

    def forecasts(self, p, theta, speed, l1, l2, rows, c, res, rng):
        """Each fish's expected heading LEAD steps after the forecast origin, from the twin (REPS replicas)."""
        s = rows - LEAD
        m = np.empty((len(rows), theta.shape[1], 2))
        for a in range(0, len(rows), 1000):
            idx = s[a:a + 1000]
            st = tuple(np.repeat(v[idx][:, None], REPS, 1) for v in (p, theta, speed, l1, l2))
            for _ in range(LEAD):
                st = self.step(st, c, res, rng)
            m[a:a + 1000] = unit(st[1]).mean(1)
        return m


def _time_mask(rows, mask, t):
    """Row mask -> time mask (each time step takes the split of the nearest target row)."""
    out = np.empty(t, bool)
    out[rows] = mask
    out[:rows[0]] = mask[0]
    out[rows[-1] + 1:] = mask[-1]
    return out


def decompose(p, theta, speed, tank, rng, true_twin=None):
    """All shares on phi^2: memory twin and fitted full twin, interleaved and chronological; plus the true twin."""
    rows, splits = layout(len(theta))
    x, l1, l2, ok = tank.design(p, theta, speed)
    mem = memory_twin(theta, rows)
    out = {"phi_mean": float((np.sqrt((unit(theta).sum(1) ** 2).sum(-1)) / theta.shape[1]).mean())}
    for name, split in splits.items():
        out[f"memory_{name}"] = share(theta, rows, split, [mem] * len(split))
        full = []
        for tr, _ in split:
            c, res = tank.fit(x, l1, ok, _time_mask(rows, tr, len(theta)))
            full.append(tank.forecasts(p, theta, speed, l1, l2, rows, c, res, rng))
        out[f"full_{name}"] = share(theta, rows, split, full)
        if true_twin is not None:
            m = tank.forecasts(p, theta, speed, l1, l2, rows, *true_twin, rng)
            out[f"true_{name}"] = share(theta, rows, split, [m] * len(split))
    return out


# ---------------------------------------------------------------- simulated fish with a known answer

def _rule(rule, k, p, th, old):
    """Interaction rules other than the fitted model's 4-nearest-neighbour terms."""
    d = np.hypot(*(p[None] - p[:, None]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    if rule == "metric":
        a, c = d < 20, d < 40
        al = np.where(a.any(1), np.angle((np.exp(1j * th)[None] * a).sum(1)) - th, 0)
        cen = (p[None] * c[..., None]).sum(1) / np.maximum(c.sum(1), 1)[:, None] - p
        at = np.where(c.any(1), np.sin(np.arctan2(cen[:, 1], cen[:, 0]) - th), 0)
        return k * np.sin(al) + 0.5 * k * at
    nn = np.argsort(d, 1)[:, :7]
    al = np.angle(np.exp(1j * old[nn]).sum(1)) - th
    cen = p[nn].mean(1) - p
    return k * np.sin(al) + 0.5 * k * np.sin(np.arctan2(cen[:, 1], cen[:, 0]) - th)


def simulate(coef, res, speed, p0, th0, rule, k, rng, steps=SIM_STEPS):
    """e32's fitted own model with interaction by `rule` ('model': its own social terms; 'loners': none;
    'metric': align within 20 cm, attract within 40 cm; 'delayed': align with the 7 nearest fish's headings
    0.5 s earlier)."""
    n = len(speed)
    p, th, l1, l2 = p0.copy(), th0.copy(), np.zeros(n), np.zeros(n)
    ps, ths = np.empty((steps, n, 2)), np.empty((steps, n))
    for t in range(steps):
        ps[t], ths[t] = p, th
        turn = e32.own_turn(p, th, l1, l2, coef) + rng.choice(res, n)
        if rule == "model":
            turn = turn + social_terms(p[None], th[None])[0] @ coef[e32.N_OWN:]
        elif rule != "loners":
            turn = turn + _rule(rule, k, p, th, ths[max(t - 5, 0)])
        p, th = e32.advance(p, th, speed, turn)
        l1, l2 = turn, l1
    return ps, ths


def _sim(job):
    n, rule, k, seed = job
    rng = np.random.default_rng([SEED, n, seed, int(k * 10), len(rule)])
    p, th, sp = kinematics(_load_adult(n, 0))
    sp = np.where(sp > 100.0, 0.0, sp)
    coef, res, _ = e32.fit_model(p, th, sp, int(TRAIN * len(th)))
    speed = np.median(sp, 0)
    ps, ths = simulate(coef, res, speed, p[0], th[0], rule, k, rng)
    spd = np.broadcast_to(speed, ths.shape).copy()
    true = (coef[:e32.N_OWN], res)
    out = decompose(ps, ths, spd, Tank("square"), rng, true_twin=true)
    a = e33.analyse(ps, ths, spd, rng, true_twin=true)
    return {"n": n, "rule": rule, "k": k, "seed": seed, **out, "proj_fit": a["S_proj"],
            "proj_true": a["true_twin"]["S_proj"]}


def _load_adult(n, rec):
    from experiments.fish_data import load
    return load(n, rec)


# ---------------------------------------------------------------- real schools

def _adult(job):
    n, rec, kind = job
    rng = np.random.default_rng([SEED, n, rec, kind == "virtual"])
    pos = _load_adult(n, rec)
    p, th, sp = kinematics(virtual(pos) if kind == "virtual" else pos)
    return {"set": "adult", "n": n, "rec": rec, "kind": kind,
            **decompose(p, th, np.where(sp > 100.0, 0.0, sp), Tank("square"), rng)}


def _zebra(job):
    n, rec, kind = job
    rng = np.random.default_rng([SEED, n, rec, kind == "virtual", 38])
    if kind == "real":
        pos, radius, dt = e38.load(n, rec)
    else:
        loaded = [e38.load(n, r) for r in e38.RECS]
        dt = float(np.mean([l[2] for l in loaded]))
        radius = float(np.mean([l[1] for l in loaded]))
        length = min(len(l[0]) for l in loaded)
        per = n // 3 + 1
        pos = np.empty((length, n, 2))
        for i in range(n):
            pos[:, i] = np.roll(loaded[(i + rec) % 3][0][:length, i], (i // 3) * length // per, axis=0)
    p, th, sp = e38.kinematics(pos, dt)
    return {"set": "zebrafish", "n": n, "rec": rec, "kind": kind,
            **decompose(p, th, e38._clean(sp), Tank("round", radius, dt), rng)}


def published(r):
    """e33 / e38 S_proj for the same group (e68.measured); zebrafish virtual groups are indexed by m = rec."""
    from experiments.e68_fish_memory_gap import measured
    adult, zr, zv = measured()
    if r["set"] == "adult":
        return adult[(r["n"], r["rec"])][r["kind"] == "virtual"]
    return (zr if r["kind"] == "real" else zv)[(r["n"], r["rec"])]


# ---------------------------------------------------------------- predictions

def predictions(res):
    sim = res["sim"]
    lon = [r for r in sim if r["rule"] == "loners"]
    sch = [r for r in sim if r["rule"] != "loners"]
    keys = [f"{t}_{s}" for t in ("memory", "full", "true") for s in ("interleaved", "chrono")]
    a1 = max(abs(r[k]) for r in lon for k in keys)
    a2 = max(abs(r["full_interleaved"] - r["full_chrono"]) for r in sch)
    a3 = sum(abs(r["true_chrono"] - r["proj_true"]) <= 0.05 for r in sch)
    a4 = max(r["memory_interleaved"] - r["true_interleaved"] for r in sch)
    a5 = sum(abs(r["full_interleaved"] - r["true_interleaved"]) <= 0.03 for r in sch)
    real = [r for r in res["real"] if r["kind"] == "real"]
    virt = [r for r in res["real"] if r["kind"] == "virtual"]
    pub = np.array([published(r) for r in real])
    col = lambda k, rs=real: np.array([r[k] for r in rs])  # noqa: E731
    b1 = float(np.median(np.abs(col("full_chrono") - pub)))
    b2 = sum(abs(r["full_interleaved"]) <= 0.1 for r in virt)
    b3 = float(np.median(col("full_interleaved") - col("full_chrono")))
    b4 = float(np.median(col("memory_interleaved") - col("full_interleaved")))
    b5 = float(np.median(np.abs(col("memory_chrono") - pub)))
    ns = len(sch)
    return [
        ("A1", "simulated loners: every share within +-0.03", a1 <= 0.03, f"max |S| {a1:.3f}"),
        ("A2", "stationary schools: chronological and interleaved splits agree within 0.03 (full twin)",
         a2 <= 0.03, f"max |diff| {a2:.3f}"),
        ("A3", "bridge: true twin on phi^2 (chrono) within 0.05 of e33's S_proj in >= 80% of schools",
         a3 >= 0.8 * ns, f"{a3}/{ns}"),
        ("A4", "memory gap not above the true twin: S_mem <= S_true + 0.05 in every school", a4 <= 0.05,
         f"max excess {a4:+.3f}"),
        ("A5", "fitted full twin within 0.03 of the true twin in >= 80% of schools", a5 >= 0.8 * ns, f"{a5}/{ns}"),
        ("B1", "real bridge: full twin on phi^2 (chrono) within 0.05 of published S_proj (median, 17 schools)",
         b1 <= 0.05, f"median |diff| {b1:.3f}"),
        ("B2", "real virtual groups: |S_full| <= 0.1 in >= 80%", b2 >= 0.8 * len(virt), f"{b2}/{len(virt)}"),
        ("B3", "drift: the chronological split lowers the share, median (interleaved - chrono) >= 0.05",
         b3 >= 0.05, f"median {b3:+.3f}"),
        ("B4", "twin content small: median (memory - full twin), interleaved, <= 0.05", b4 <= 0.05,
         f"median {b4:+.3f}"),
        ("B5", "correction: the memory gap scored on the chronological split, median |S - S_proj| <= 0.07",
         b5 <= 0.07, f"median |diff| {b5:.3f} (e72: 0.140)"),
    ]


def analyze(res) -> str:
    cols = ("memory_interleaved", "full_interleaved", "memory_chrono", "full_chrono")
    lines = ["Experiment 77: why the quadratic memory gap runs above the fitted fish twins", "",
             "  Simulated schools (square tank, model fitted to adult recording 0):",
             "  N   rule      k    seed  phi   mem/int full/int true/int mem/chr full/chr true/chr e33 fit e33 true"]
    for r in res["sim"]:
        lines.append(f"  {r['n']:<3d} {r['rule']:8} {r['k']:<4} {r['seed']}  {r['phi_mean']:.2f}  " + "  ".join(
            f"{r[k]:+.3f}" for k in ("memory_interleaved", "full_interleaved", "true_interleaved", "memory_chrono",
                                     "full_chrono", "true_chrono", "proj_fit", "proj_true")))
    lines += ["", "  Real schools and virtual groups (published = e33 / e38 S_proj):",
              "  set        group         kind      phi   mem/int full/int mem/chr full/chr published"]
    for r in res["real"]:
        lines.append(f"  {r['set']:10} N{r['n']:<3d} rec{r['rec']}     {r['kind']:8}  {r['phi_mean']:.2f}  "
                     + "  ".join(f"{r[k]:+.3f}" for k in cols) + f"  {published(r):+.3f}")
    real = [r for r in res["real"] if r["kind"] == "real"]
    if real:
        pub = np.array([published(r) for r in real])
        c = {k: np.array([r[k] for r in real]) for k in cols}
        lines += ["", "  Decomposition of e72's excess over the published share (medians over real schools):",
                  f"    total  memory/interleaved - published   {np.median(c['memory_interleaved'] - pub):+.3f}",
                  f"    twin   memory - full twin (interleaved)  {np.median(c['memory_interleaved'] - c['full_interleaved']):+.3f}",
                  f"    split  full twin, interleaved - chrono   {np.median(c['full_interleaved'] - c['full_chrono']):+.3f}",
                  f"    law    full twin chrono - published      {np.median(c['full_chrono'] - pub):+.3f}"]
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from experiments.fish_data import RECORDINGS
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    sims = [(n, r, k, s) for n in (8, 39) for r, k in (("loners", 0.0),) + RULES for s in SIM_SEEDS]
    adult = [(n, r, k) for n, recs in RECORDINGS.items() for r in recs for k in ("real", "virtual")]
    zebra = [(n, r, "real") for n in e38.SIZES for r in e38.RECS] + [(n, m, "virtual") for n in e38.SIZES
                                                                     for m in range(3)]
    if quick:
        sims, adult, zebra = sims[:2], adult[:2], zebra[:2]
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = {"sim": list(pool.map(_sim, sims)),
               "real": list(pool.map(_adult, adult)) + list(pool.map(_zebra, zebra))}
    (out_dir / "e77_fish_gap_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e77_figures import make_figures
    make_figures(res, out_dir / "e77_figures", text)
    return text


if __name__ == "__main__":
    main()
