"""Experiment 62: the interaction share of brain areas, with a repaired twin.

e61 scored 4/9, and its real-data numbers are void. Three faults, all found before any of e62 ran:
  1. The per-neuron Poisson fits had no step control. For some low-rate neurons Newton's method
     diverged to the clip (403 spikes per 50 ms against a true 0.2), and the summed twin forecast
     was wrecked: real and virtual groups alike reached S_proj 0.9 or more.
  2. The twin's own-history features (log counts) were not the features the group law uses (raw
     counts and the latest window squared). The projected share is zero without interaction only
     if each neuron's residual is uncorrelated with those features. With a slow shared drive,
     the simulated virtual groups scored 0.06-0.10 instead of 0.
  3. The simulated coupling was too weak (pairwise correlation 0.01) to shape the population
     count at all (S_proj <= 0.003). So N2/N5 tested nothing. That weak coupling is kept here, as
     a reported condition.
Repairs:
  * Newton steps with backtracking on the penalized log-likelihood; non-converged fits counted.
  * Twin features: each neuron's own counts in the last 5, 5-10, 10-20, 20-30, 30-50 bins, the latest
    squared, and the log of each window, plus the environment. The group law uses the area's own
    five windows and the latest squared, plus the environment. Forecasts start at bin 49.
  * E1 also includes the time in the session (six bumps over the trial index), a recorded cause:
    whole areas drift by 30% over a session.
  * Strong coupling: mean weight 0.12 instead of 0.06 (pairwise correlation 0.02-0.04, set on
    dynamics only).
Everything else is as in e61: 50 ms horizon, environments E0 / E1 / E2, test blocks, virtual groups
permuted within stimulus condition, 400 trial bootstraps.

Predictions, fixed before running (N3 and N4 were seen in e61 and are replications):
  N1  Independent networks: |S_proj| <= 0.02 in 5/5.
  N2  Strongly coupled networks: S_proj >= 0.05 with lower bound > 0 in 5/5; virtual |S| <= 0.02.
  N3  Weakly coupled networks (e61's): S_proj <= 0.02 in >= 4/5 (weak random coupling leaves the
      population count unshaped).
  N4  Hidden shared drive is flagged: S_proj >= 0.05 in >= 4/5.
  N5  The repaired twin cleans the hidden drive's virtual groups: |S_virtual| <= 0.02 in >= 4/5.
  N6  Recorded drive: |S_proj| <= 0.02 in >= 4/5.
  N7  Strong coupling with the drive recorded: S_proj >= 0.05 in 5/5.
  R0  The fits converge for >= 99% of real neurons.
  R1  Real virtual groups are clean: |S_virtual| <= 0.03 in >= 80% of areas.
  R2  Coordination beyond the stimulus: S_proj(E0) > S_virtual + 0.02, lower bound > S_virtual,
      in >= 80% of areas.
  R3  Recorded causes absorb part of it: median S_proj(E2) < median S_proj(E0).
  R4  Coordination beyond all recorded causes: S_proj(E2) lower bound > 0 in >= 50% of areas.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e61_neural_share import _bumps, _cs, _standardize, _warp, ridge, split, virtual_perm

SEED = 62
NB = 250
STIM = 50
H = 5
T_FIRST = 49
STRIDE = 2
HIST = ((0, 5), (5, 10), (10, 20), (20, 30), (30, 50))
N_BOOT = 400
MIN_RATE = 0.5
MIN_NEURONS = 30
SIM_SEEDS = (0, 1, 2, 3, 4)
CONDITIONS = ("independent", "coupled", "weak", "hidden", "recorded", "coupled_rec")
COUPLING = {"coupled": 0.12, "coupled_rec": 0.12, "weak": 0.06}
TS = np.arange(T_FIRST, NB - H, STRIDE)


# ---------------------------------------------------------------- features

def windows(x):
    c = _cs(x)
    hist = np.stack([c[..., TS + 1 - a] - c[..., TS + 1 - b] for a, b in HIST], -1)
    return hist, c[..., TS + 1 + H] - c[..., TS + 1]


def own(hist):
    """Twin history features: raw windows, latest squared, log windows."""
    return np.concatenate([hist, hist[..., :1] ** 2, np.log1p(hist)], -1)


def group(hist):
    """Group-law history features: raw windows, latest squared."""
    return np.concatenate([hist, hist[..., :1] ** 2], -1)


def env_stim(cl, cr):
    u = (TS + H / 2 - STIM)[None, :].repeat(len(cl), 0)
    b = _bumps(_warp(u), np.linspace(_warp(-25), _warp(195), 12))
    return np.concatenate([b, b * cl[:, None, None], b * cr[:, None, None]], -1)


def env_task(td):
    nt = len(td["cl"])
    out = []
    for key in ("gocue", "feedback"):
        u = TS[None, :] + H / 2 - STIM - td[key][:, None] / 0.01
        out.append(_bumps(u, (-15, 0, 10, 25, 50, 90)))
    c = {k: _cs(np.abs(td[k]) if k == "wheel" else td[k]) for k in ("wheel", "face", "licks")}
    beh = [(c[k][:, TS + 1] - c[k][:, TS - 4]) / 5 for k in ("wheel", "face", "licks")]
    beh.append(td["pupil"][:, TS])
    out.append(np.stack(beh, -1))
    drift = _bumps(np.arange(nt, dtype=float), np.linspace(0, nt - 1, 6))
    out.append(np.repeat(drift[:, None, :], len(TS), 1))
    return np.concatenate(out, -1)


def env_areas(counts_by_area):
    out = []
    for x in counts_by_area:
        c = _cs(x)
        out += [np.log1p(c[:, TS + 1] - c[:, TS - 4]), np.log1p(c[:, TS - 4] - c[:, TS - 19])]
    return np.stack(out, -1)


# ---------------------------------------------------------------- models

def poisson_glm(x, y, train, lam=10.0, iters=30):
    """Ridge Poisson regression by damped Newton steps (backtracking). Returns (means, converged)."""
    x = np.column_stack([np.ones(len(x)), _standardize(x, train)])
    xt, yt = x[train], y[train]
    pen = np.full(x.shape[1], lam)
    pen[0] = 0.0

    def obj(w):
        eta = np.clip(xt @ w, -20, 8)
        return float(yt @ eta - np.exp(eta).sum() - 0.5 * (pen * w * w).sum())

    w = np.zeros(x.shape[1])
    w[0] = np.log(max(yt.mean(), 1e-3))
    f = obj(w)
    converged = False
    for _ in range(iters):
        mu = np.exp(np.clip(xt @ w, -20, 8))
        grad = xt.T @ (yt - mu) - pen * w
        hess = (xt * mu[:, None]).T @ xt + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-4:
            f_new = obj(w + t * step)
            if f_new >= f - 1e-10 * abs(f):
                break
            t /= 2
        w, f_old, f = w + t * step, f, f_new
        if abs(f - f_old) <= 1e-9 * abs(f) + 1e-12 or np.max(np.abs(t * step)) < 1e-7:
            converged = True
            break
    return np.exp(np.clip(x @ w, -20, 8)), converged


def twin(x_counts, env, train_tr):
    hist, target = windows(x_counts)
    n, nt, nr = target.shape
    train = np.repeat(np.isin(np.arange(nt), train_tr), nr)
    envf = env.reshape(nt * nr, -1)
    pred = np.empty_like(target)
    conv = 0
    for i in range(n):
        xi = np.column_stack([own(hist[i]).reshape(nt * nr, -1), envf])
        p, ok = poisson_glm(xi, target[i].ravel(), train)
        pred[i] = p.reshape(nt, nr)
        conv += ok
    return pred, conv


def share(phi_hist, phi, g0, env, train_tr, test_tr, rng):
    nt, nr = phi.shape
    x = np.column_stack([group(phi_hist).reshape(nt * nr, -1), env.reshape(nt * nr, -1)])
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


def analyse_group(x, envs, cl, cr, rng):
    nt = x.shape[1]
    train_tr, test_tr = split(nt)
    phi_hist, phi = windows(x.sum(0))
    out = {}
    preds = {}
    for name, env in envs.items():
        preds[name], conv = twin(x, env, train_tr)
        out[name] = share(phi_hist, phi, preds[name].sum(0), env, train_tr, test_tr, rng)
        out[name]["converged"] = int(conv)
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
    mean_w = COUPLING.get(cond, 0.0)
    drive = cond in ("hidden", "recorded", "coupled_rec")
    w = np.where(rng.random((n, n)) < 0.2, mean_w + 0.12 * rng.normal(size=(n, n)), 0.0) * (mean_w > 0)
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
            if mean_w > 0:
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
    w = x[:, :, 60:].reshape(len(x), x.shape[1], -1, 5).sum(-1).reshape(len(x), -1).astype(float)
    cc = np.corrcoef(w)
    corr = float(np.nanmean(cc[np.triu_indices(len(x), 1)]))
    return {"cond": cond, "seed": seed, "rate": float(x.mean() / 0.01), "corr": corr, **res}


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
    ns = len(SIM_SEEDS)
    n1 = sum(abs(v) <= 0.02 for v in s(sim["independent"]))
    n2 = sum(r["E0"]["S"] >= 0.05 and r["E0"]["lo"] > 0 for r in sim["coupled"])
    n2v = sum(abs(v) <= 0.02 for v in s(sim["coupled"], "virtual"))
    n3 = sum(v <= 0.02 for v in s(sim["weak"]))
    n4 = sum(v >= 0.05 for v in s(sim["hidden"]))
    n5 = sum(abs(v) <= 0.02 for v in s(sim["hidden"], "virtual"))
    n6 = sum(abs(v) <= 0.02 for v in s(sim["recorded"]))
    n7 = sum(v >= 0.05 for v in s(sim["coupled_rec"]))
    real = res["real"]
    m = len(real)
    neurons = sum(3 * r["n"] for r in real)
    conv = sum(r[k]["converged"] for r in real for k in ("E0", "E1", "E2"))
    r1 = sum(abs(r["virtual"]["S"]) <= 0.03 for r in real)
    r2 = sum(r["E0"]["S"] > r["virtual"]["S"] + 0.02 and r["E0"]["lo"] > r["virtual"]["S"] for r in real)
    med0 = float(np.median([r["E0"]["S"] for r in real])) if real else np.nan
    med2 = float(np.median([r["E2"]["S"] for r in real])) if real else np.nan
    r4 = sum(r["E2"]["lo"] > 0 for r in real)
    return [
        ("N1", "independent networks: |S_proj| <= 0.02 in 5/5", n1 == ns, f"{n1}/{ns}"),
        ("N2", "strongly coupled networks: S_proj >= 0.05 (lower > 0) in 5/5; virtual |S| <= 0.02",
         n2 == ns and n2v == ns, f"{n2}/{ns}, virtual {n2v}/{ns}"),
        ("N3", "weakly coupled networks (e61's): S_proj <= 0.02 in >= 4/5", n3 >= 4, f"{n3}/{ns}"),
        ("N4", "hidden shared drive is flagged: S_proj >= 0.05 in >= 4/5", n4 >= 4, f"{n4}/{ns}"),
        ("N5", "repaired twin cleans the hidden drive's virtual groups: |S_virtual| <= 0.02 in >= 4/5", n5 >= 4,
         f"{n5}/{ns}"),
        ("N6", "recorded drive: |S_proj| <= 0.02 in >= 4/5", n6 >= 4, f"{n6}/{ns}"),
        ("N7", "strong coupling with the drive recorded: S_proj >= 0.05 in 5/5", n7 == ns, f"{n7}/{ns}"),
        ("R0", "fits converge for >= 99% of real neurons", conv >= 0.99 * neurons, f"{conv}/{neurons}"),
        ("R1", "real virtual groups clean: |S_virtual| <= 0.03 in >= 80% of areas", r1 >= 0.8 * m, f"{r1}/{m}"),
        ("R2", "coordination beyond the stimulus: S(E0) > S_virtual + 0.02, lower > S_virtual, >= 80%",
         r2 >= 0.8 * m, f"{r2}/{m}"),
        ("R3", "recorded causes absorb part: median S(E2) < median S(E0)", med2 < med0,
         f"median E0 {med0:+.3f}, E2 {med2:+.3f}"),
        ("R4", "coordination beyond all recorded causes: S(E2) lower > 0 in >= 50%", r4 >= 0.5 * m, f"{r4}/{m}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 62: the interaction share of brain areas, with a repaired twin", "",
             "  Simulated networks (60 neurons, S_proj [90% interval], virtual):"]
    for r in res["sim"]:
        lines.append(f"  {r['cond']:12} seed {r['seed']}  rate {r['rate']:5.1f} Hz  corr {r['corr']:.3f}  "
                     f"S {r['E0']['S']:+.3f} [{r['E0']['lo']:+.3f}, {r['E0']['hi']:+.3f}]  "
                     f"virtual {r['virtual']['S']:+.3f}")
    lines += ["", "  Real areas (S_proj with stimulus / + task, behaviour, session time / + other areas; virtual):"]
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
    (out_dir / "e62_neural_share_repaired_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e62_figures import make_figures
    make_figures(res, out_dir / "e62_figures", text)
    return text


if __name__ == "__main__":
    main()
