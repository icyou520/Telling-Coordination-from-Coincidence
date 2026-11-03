"""Experiment 61: how much of a brain area's activity is shaped by its neurons interacting?

The interaction share, moved to a new layer: neurons -> the population activity of a brain area.

Group quantity phi: the total spike count of an area's neurons in the next 50 ms (5 bins of 10 ms),
forecast at every second bin of a trial from bin 29 on. Recorded common causes ("environment") come
in three nested sets:
  E0  stimulus: 12 time-in-trial bumps (warped time since stimulus onset), alone and times each
      side's contrast;
  E1  E0 + task events and behaviour: bumps around the go cue and feedback times, wheel speed, pupil
      area, face motion and licks at the forecast time (never later: the area may drive them);
  E2  E1 + the recent total activity of every other area of the session with >= 10 neurons.
Twin: one Poisson GLM per neuron (log link, ridge), from its own history (counts in the last 5, 5-10,
10-20, 20-30 bins) and the environment, with no other neuron. Twin forecast g0 = sum of the neurons'
predicted counts. Real law G and the twin's projected law G0_proj: the same ridge regression of phi or
of g0 on the area's own history (the same four windows, the latest one squared) and the environment,
fitted on training trials. S_proj = 1 - MSE(G) / MSE(G0_proj) on test trials (blocks of 10 trials,
every third block held out); 90% interval from 400 bootstrap resamples of test trials. Without
interaction (and a correct twin) G and G0_proj estimate the same conditional mean, so S_proj = 0.

Virtual groups: each neuron's trials are permuted within its stimulus condition (left, right contrast),
separately in training and test trials; this keeps every neuron's stimulus-driven activity and its
twin, and breaks all trial-to-trial coordination. Uses E0 (the only environment shared by a virtual
group).

Part A, simulated Poisson networks (ground truth), 60 neurons, 300 trials of the same layout:
  independent   no coupling, no shared drive;
  coupled       sparse coupling between neurons (the last two bins of input);
  hidden        no coupling, a slow shared drive (OU, tau 300 ms) that is not recorded;
  recorded      as hidden, but the drive's current value is part of the environment;
  coupled_rec   coupling and the drive, drive recorded.
Five seeds each.

Part B, real data: Steinmetz et al. (2019), 39 sessions; every area with >= 30 neurons firing
>= 0.5 Hz (root excluded).

Predictions, fixed before running:
  N1  Independent networks: |S_proj| <= 0.02 in 5/5.
  N2  Coupled networks: S_proj >= 0.05 with lower bound > 0 in 5/5; their virtual groups |S| <= 0.02.
  N3  The known confound: hidden shared drive gives S_proj >= 0.05 in >= 4/5 (the method alone cannot
      tell an unrecorded common drive from coupling).
  N4  Recorded drive: |S_proj| <= 0.02 in >= 4/5.
  N5  Coupling is still found with the drive recorded: coupled_rec S_proj >= 0.05 in 5/5.
  R1  Real virtual groups are clean: |S_virtual| <= 0.03 in >= 80% of areas.
  R2  Real areas show coordination beyond the stimulus: S_proj(E0) > S_virtual + 0.02 with lower
      bound > S_virtual in >= 80% of areas.
  R3  Recorded causes absorb part of it: median S_proj(E2) < median S_proj(E0).
  R4  Coordination beyond all recorded causes: S_proj(E2) lower bound > 0 in >= 50% of areas.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEED = 61
NB = 250
STIM = 50
H = 5
T_FIRST = 29
STRIDE = 2
HIST = ((0, 5), (5, 10), (10, 20), (20, 30))
N_BOOT = 400
MIN_RATE = 0.5
MIN_NEURONS = 30
SIM_SEEDS = (0, 1, 2, 3, 4)
CONDITIONS = ("independent", "coupled", "hidden", "recorded", "coupled_rec")
TS = np.arange(T_FIRST, NB - H, STRIDE)


# ---------------------------------------------------------------- features

def _cs(x):
    """Cumulative sums over bins with a leading zero: sum of bins [u, v) = c[..., v] - c[..., u]."""
    return np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, -1, dtype=float)], -1)


def windows(x):
    """History windows (..., trials, rows, 4) and the next-H target (..., trials, rows) of counts x."""
    c = _cs(x)
    hist = np.stack([c[..., TS + 1 - a] - c[..., TS + 1 - b] for a, b in HIST], -1)
    return hist, c[..., TS + 1 + H] - c[..., TS + 1]


def _bumps(u, centers):
    centers = np.asarray(centers, float)
    width = np.gradient(centers)
    return np.exp(-0.5 * ((u[..., None] - centers) / width) ** 2)


def _warp(u):
    return np.sign(u) * np.log1p(np.abs(u) / 5)


def env_stim(cl, cr):
    """E0: (trials, rows, 36)."""
    u = (TS + H / 2 - STIM)[None, :].repeat(len(cl), 0)
    b = _bumps(_warp(u), np.linspace(_warp(-25), _warp(195), 12))
    return np.concatenate([b, b * cl[:, None, None], b * cr[:, None, None]], -1)


def env_task(td):
    """E1 additions: go-cue and feedback bumps, behaviour at the forecast time."""
    out = []
    for key in ("gocue", "feedback"):
        u = TS[None, :] + H / 2 - STIM - td[key][:, None] / 0.01
        out.append(_bumps(u, (-15, 0, 10, 25, 50, 90)))
    c = {k: _cs(np.abs(td[k]) if k == "wheel" else td[k]) for k in ("wheel", "face", "licks")}
    beh = [(c[k][:, TS + 1] - c[k][:, TS - 4]) / 5 for k in ("wheel", "face", "licks")]
    beh.append(td["pupil"][:, TS])
    out.append(np.stack(beh, -1))
    return np.concatenate(out, -1)


def env_areas(counts_by_area):
    """E2 additions: log activity of each other area in the last 5 and 5-20 bins."""
    out = []
    for x in counts_by_area:
        c = _cs(x)
        out += [np.log1p(c[:, TS + 1] - c[:, TS - 4]), np.log1p(c[:, TS - 4] - c[:, TS - 19])]
    return np.stack(out, -1)


# ---------------------------------------------------------------- models

def _standardize(x, train):
    mu, sd = x[train].mean(0), x[train].std(0)
    return (x - mu) / np.where(sd > 1e-9, sd, 1.0)


def poisson_glm(x, y, train, lam=10.0, iters=12):
    """Ridge Poisson regression (log link) fitted by Newton steps on train; predicted means on all rows."""
    x = np.column_stack([np.ones(len(x)), _standardize(x, train)])
    xt, yt = x[train], y[train]
    w = np.zeros(x.shape[1])
    w[0] = np.log(max(yt.mean(), 1e-3))
    pen = np.full(x.shape[1], lam)
    pen[0] = 0.0
    for _ in range(iters):
        mu = np.exp(np.clip(xt @ w, -20, 6))
        grad = xt.T @ (yt - mu) - pen * w
        hess = (xt * mu[:, None]).T @ xt + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.max(np.abs(step)) < 1e-6:
            break
    return np.exp(np.clip(x @ w, -20, 6))


def ridge(x, y, train, lam=1.0):
    x = np.column_stack([np.ones(len(x)), _standardize(x, train)])
    pen = np.full(x.shape[1], lam)
    pen[0] = 0.0
    w = np.linalg.solve(x[train].T @ x[train] + np.diag(pen), x[train].T @ y[train])
    return x @ w


def share(phi_hist, phi, g0, env, train_tr, test_tr, rng):
    """S_proj with a trial-bootstrap interval. Arrays are (trials, rows, ...)."""
    nt, nr = phi.shape
    h = phi_hist.reshape(nt * nr, -1)
    x = np.column_stack([h, h[:, 0] ** 2, env.reshape(nt * nr, -1)])
    train = np.repeat(np.isin(np.arange(nt), train_tr), nr)
    y, y0 = phi.ravel(), g0.ravel()
    e_g = ((y - ridge(x, y, train)) ** 2).reshape(nt, nr).sum(1)
    e_0 = ((y - ridge(x, y0, train)) ** 2).reshape(nt, nr).sum(1)
    s = 1 - e_g[test_tr].sum() / e_0[test_tr].sum()
    boot = []
    for _ in range(N_BOOT):
        b = rng.choice(test_tr, len(test_tr))
        boot.append(1 - e_g[b].sum() / e_0[b].sum())
    lo, hi = np.quantile(boot, [0.05, 0.95])
    return {"S": float(s), "lo": float(lo), "hi": float(hi)}


def twin(x_counts, env, train_tr):
    """Per-neuron twin predictions (neurons, trials, rows) and the neurons' history windows."""
    hist, target = windows(x_counts)
    n, nt, nr = target.shape
    train = np.repeat(np.isin(np.arange(nt), train_tr), nr)
    envf = env.reshape(nt * nr, -1)
    pred = np.empty_like(target)
    for i in range(n):
        xi = np.column_stack([np.log1p(hist[i].reshape(nt * nr, -1)), envf])
        pred[i] = poisson_glm(xi, target[i].ravel(), train).reshape(nt, nr)
    return pred


def split(nt):
    test = (np.arange(nt) // 10) % 3 == 2
    return np.flatnonzero(~test), np.flatnonzero(test)


def virtual_perm(cl, cr, trials, rng):
    """A permutation of the given trials within each stimulus condition."""
    perm = np.arange(len(cl))
    for c in set(zip(cl[trials], cr[trials])):
        idx = trials[(cl[trials] == c[0]) & (cr[trials] == c[1])]
        perm[idx] = rng.permutation(idx)
    return perm


def analyse_group(x, envs, cl, cr, rng):
    """S_proj for each environment set and for virtual groups (E0). x: (neurons, trials, bins)."""
    nt = x.shape[1]
    train_tr, test_tr = split(nt)
    phi_hist, phi = windows(x.sum(0))
    out = {}
    preds = {}
    for name, env in envs.items():
        preds[name] = twin(x, env, train_tr)
        out[name] = share(phi_hist, phi, preds[name].sum(0), env, train_tr, test_tr, rng)
    perms = []
    for _ in range(len(x)):
        p = virtual_perm(cl, cr, train_tr, rng)
        p[test_tr] = virtual_perm(cl, cr, test_tr, rng)[test_tr]
        perms.append(p)
    xv = np.stack([x[i][p] for i, p in enumerate(perms)])
    gv = np.stack([preds["E0"][i][p] for i, p in enumerate(perms)]).sum(0)
    vh, v = windows(xv.sum(0))
    out["virtual"] = share(vh, v, gv, envs["E0"], train_tr, test_tr, rng)
    return out


# ---------------------------------------------------------------- part A: simulated networks

def simulate(cond, seed, n=60, trials=300):
    rng = np.random.default_rng([SEED, CONDITIONS.index(cond), seed])
    cl = rng.choice([0, 0.25, 0.5, 1.0], trials)
    cr = rng.choice([0, 0.25, 0.5, 1.0], trials)
    base = np.log(rng.uniform(2, 15, n) * 0.01)
    lat = rng.uniform(5, 15, n)
    t = np.arange(NB) - STIM
    kern = np.where(t[None] > 0, (t[None] / lat[:, None]) * np.exp(1 - t[None] / lat[:, None]), 0.0)
    side = rng.normal(0, 1, (n, 2))
    stim = kern[:, None] * (np.abs(side[:, :1, None]) * 0.3 + side[:, 0, None, None] * cl[None, :, None]
                            + side[:, 1, None, None] * cr[None, :, None])
    coupled = cond in ("coupled", "coupled_rec")
    drive = cond in ("hidden", "recorded", "coupled_rec")
    w = np.where(rng.random((n, n)) < 0.2, 0.06 + 0.12 * rng.normal(size=(n, n)), 0.0) * coupled
    np.fill_diagonal(w, 0.0)
    load = rng.normal(0.4, 0.15, n) * drive
    z = np.zeros((trials, NB))
    for k in range(1, NB):
        z[:, k] = z[:, k - 1] * (1 - 0.01 / 0.3) + np.sqrt(2 * 0.01 / 0.3) * rng.normal(size=trials)
    x = np.zeros((n, trials, NB), dtype=np.int16)
    for k in range(NB):
        eta = base[:, None] + stim[:, :, k] + load[:, None] * z[None, :, k]
        if k >= 1:
            eta = eta - 2.0 * x[:, :, k - 1]
            if coupled:
                eta = eta + w @ (x[:, :, k - 1] + (x[:, :, k - 2] if k >= 2 else 0))
        x[:, :, k] = rng.poisson(np.exp(np.minimum(eta, 0.0)))
    return x, cl, cr, z


def _sim_run(job):
    cond, seed = job
    x, cl, cr, z = simulate(cond, seed)
    env = env_stim(cl, cr)
    if cond in ("recorded", "coupled_rec"):
        env = np.concatenate([env, z[:, TS, None]], -1)
    res = analyse_group(x, {"E0": env}, cl, cr, np.random.default_rng([SEED, 1, seed]))
    rate = float(x.mean() / 0.01)
    return {"cond": cond, "seed": seed, "rate": rate, **res}


# ---------------------------------------------------------------- part B: real areas

def _session_run(s):
    from experiments import steinmetz_data as sd
    d = sd.session(s)
    td = sd.trial_data(s)
    spk = d["spks"]
    names = d["brain_area"]
    rates = spk.mean((1, 2)) / 0.01
    e0 = env_stim(td["cl"], td["cr"])
    e1 = np.concatenate([e0, env_task(td)], -1)
    labels, counts = np.unique(names, return_counts=True)
    out = []
    for area in labels:
        sel = (names == area) & (rates >= MIN_RATE)
        if area == "root" or sel.sum() < MIN_NEURONS:
            continue
        others = [spk[names == a].sum(0) for a, c in zip(labels, counts) if a not in (area, "root") and c >= 10]
        e2 = np.concatenate([e1, env_areas(others)], -1) if others else e1
        x = spk[sel].astype(np.int16)
        res = analyse_group(x, {"E0": e0, "E1": e1, "E2": e2}, td["cl"], td["cr"],
                            np.random.default_rng([SEED, 2, s, int(sel.sum())]))
        out.append({"session": s, "mouse": str(d["mouse_name"]), "area": str(area), "n": int(sel.sum()),
                    "trials": int(spk.shape[1]), "n_other": len(others), **res})
    return out


# ---------------------------------------------------------------- analysis

def predictions(res):
    sim = {c: [r for r in res["sim"] if r["cond"] == c] for c in CONDITIONS}
    s = lambda rs, k="E0": [r[k]["S"] for r in rs]  # noqa: E731
    n1 = sum(abs(v) <= 0.02 for v in s(sim["independent"]))
    n2 = sum(r["E0"]["S"] >= 0.05 and r["E0"]["lo"] > 0 for r in sim["coupled"])
    n2v = sum(abs(v) <= 0.02 for v in s(sim["coupled"], "virtual"))
    n3 = sum(v >= 0.05 for v in s(sim["hidden"]))
    n4 = sum(abs(v) <= 0.02 for v in s(sim["recorded"]))
    n5 = sum(v >= 0.05 for v in s(sim["coupled_rec"]))
    real = res["real"]
    m = len(real)
    r1 = sum(abs(r["virtual"]["S"]) <= 0.03 for r in real)
    r2 = sum(r["E0"]["S"] > r["virtual"]["S"] + 0.02 and r["E0"]["lo"] > r["virtual"]["S"] for r in real)
    med0 = float(np.median([r["E0"]["S"] for r in real])) if real else np.nan
    med2 = float(np.median([r["E2"]["S"] for r in real])) if real else np.nan
    r4 = sum(r["E2"]["lo"] > 0 for r in real)
    ns = len(SIM_SEEDS)
    return [
        ("N1", "independent networks: |S_proj| <= 0.02 in 5/5", n1 == ns, f"{n1}/{ns}"),
        ("N2", "coupled networks: S_proj >= 0.05 (lower > 0) in 5/5; virtual |S| <= 0.02",
         n2 == ns and n2v == ns, f"{n2}/{ns}, virtual {n2v}/{ns}"),
        ("N3", "hidden shared drive is flagged: S_proj >= 0.05 in >= 4/5 (the known confound)", n3 >= 4,
         f"{n3}/{ns}"),
        ("N4", "recorded drive: |S_proj| <= 0.02 in >= 4/5", n4 >= 4, f"{n4}/{ns}"),
        ("N5", "coupling found with the drive recorded: S_proj >= 0.05 in 5/5", n5 == ns, f"{n5}/{ns}"),
        ("R1", "real virtual groups clean: |S_virtual| <= 0.03 in >= 80% of areas", r1 >= 0.8 * m, f"{r1}/{m}"),
        ("R2", "coordination beyond the stimulus: S(E0) > S_virtual + 0.02, lower > S_virtual, >= 80%",
         r2 >= 0.8 * m, f"{r2}/{m}"),
        ("R3", "recorded causes absorb part: median S(E2) < median S(E0)", med2 < med0,
         f"median E0 {med0:+.3f}, E2 {med2:+.3f}"),
        ("R4", "coordination beyond all recorded causes: S(E2) lower > 0 in >= 50%", r4 >= 0.5 * m, f"{r4}/{m}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 61: how much of a brain area's activity is shaped by its neurons interacting?", "",
             "  Simulated networks (60 neurons, S_proj [90% interval], virtual):"]
    for r in res["sim"]:
        lines.append(f"  {r['cond']:12} seed {r['seed']}  rate {r['rate']:5.1f} Hz  S {r['E0']['S']:+.3f} "
                     f"[{r['E0']['lo']:+.3f}, {r['E0']['hi']:+.3f}]  virtual {r['virtual']['S']:+.3f}")
    lines += ["", "  Real areas (S_proj with stimulus / + task and behaviour / + other areas; virtual):"]
    for r in sorted(res["real"], key=lambda r: (r["area"], r["session"])):
        lines.append(f"  {r['area']:6} s{r['session']:02d} n {r['n']:3d}  E0 {r['E0']['S']:+.3f} "
                     f"[{r['E0']['lo']:+.3f}]  E1 {r['E1']['S']:+.3f}  E2 {r['E2']['S']:+.3f} "
                     f"[{r['E2']['lo']:+.3f}]  virtual {r['virtual']['S']:+.3f}")
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
    sim_jobs = [(c, s) for c in CONDITIONS for s in SIM_SEEDS]
    sessions = list(range(39))
    if quick:
        sim_jobs, sessions = sim_jobs[::5], [0]
    with ProcessPoolExecutor(max_workers=6) as pool:
        sim = list(pool.map(_sim_run, sim_jobs))
        real = [r for rs in pool.map(_session_run, sessions) for r in rs]
    res = {"sim": sim, "real": real}
    (out_dir / "e61_neural_share_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e61_figures import make_figures
    make_figures(res, out_dir / "e61_figures", text)
    return text


if __name__ == "__main__":
    main()
