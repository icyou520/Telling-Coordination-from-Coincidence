"""Experiment 64: does a brain area's interaction share survive switching off a major input?

e62 left one question open: coordination beyond recorded causes can be the neurons' interactions or an input
nobody recorded, and the interaction share alone cannot tell them apart. A perturbation can. Inagaki, Chen et al.
(2022) recorded the anterior lateral motor cortex (ALM) of mice while optogenetically silencing the midbrain
(PPN/MRN) that drives it, through the thalamus, around the go cue (light from 0.6 s before to 0.6 s after the cue).
If ALM's coordination comes from that input, its interaction share should fall when the input is off; if it is
local, it should stay.

Data: 21 sessions with >= 20 simultaneously recorded ALM units and >= 20 silenced trials (inagaki_data.load;
types 0 and 3, good trials). Spike counts in 10 ms bins, aligned to the go cue.
Windows (forecast rows every second bin; the 50 ms target and 200 ms of history inside the window):
  light     history and target within the light (0.6 s before to 0.6 s after the cue);
  baseline  the delay before the light (from 1.2 s before the cue, target ending before the light).
Measure: e62's projected share, with e62's twin (per-neuron Poisson model: own counts in the last 5, 5-10,
10-20 bins, the latest squared, logs) and group law (the ALM total's same windows, the latest squared), and an
environment of 6 time bumps over the window, alone and times the trial's instructed side. Because silenced
trials are few, every model is cross-fitted (5 folds over trials): each trial is forecast by models fitted on the
others. Control trials are subsampled to the number of silenced trials (5 draws, averaged), so both conditions
have the same power. 90% interval: 400 bootstrap resamples of trials (first draw).
Virtual groups: each neuron's trials permuted within instructed side (same condition).

Part A, simulated networks with this design (40 neurons; 40 control and 40 silenced trials; 5 seeds):
  independent  no coupling;
  coupled      e62's strong coupling; silencing removes the go-cue input (an external drive to every neuron);
  hidden       no coupling; a slow shared input (OU, tau 300 ms, e62's loading), switched off by silencing.

Predictions, fixed before running:
  A1  Independent: |S| <= 0.02 in both conditions, in >= 4/5 seeds.
  A2  The test sees an input switched off: hidden, S(light) silenced - control <= -0.1 in >= 4/5 seeds.
  A3  The test does not mistake local coupling for input: coupled, |S silenced - S control| (light)
      <= 0.03 in >= 4/5 seeds.
  R0  Silencing works: the ALM total's rise from the 300 ms before the cue to the 300 ms after is smaller in
      silenced than in control trials in >= 75% of sessions.
  R1  Virtual groups are clean: |S_virtual| <= 0.03 in >= 80% of the session x condition x window cells.
  R2  Before the light nothing differs: |median over sessions of S(baseline) silenced - control| <= 0.02.
  R3  ALM's coordination depends on the midbrain input: median over sessions of
      [S(light) silenced - control] - [S(baseline) silenced - control] < 0, sign test one-sided p < 0.05.
  R4  Coordination survives the input's removal: median S(light, silenced) > median S_virtual(light, silenced)
      + 0.02.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

import numpy as np

from experiments.e61_neural_share import _bumps, _cs, ridge
from experiments.e62_neural_share_repaired import group, own, poisson_glm

SEED = 64
H = 5
HIST = ((0, 5), (5, 10), (10, 20))
CUE = 200
WINDOWS = {"light": np.arange(160, 255, 2), "baseline": np.arange(100, 135, 2)}
FOLDS = 5
DRAWS = 5
N_BOOT = 400
MIN_UNITS = 20
MIN_SILENCED = 20
SIM_SEEDS = (0, 1, 2, 3, 4)
SIM_CONDITIONS = ("independent", "coupled", "hidden")


def windows(x, ts):
    c = _cs(x)
    hist = np.stack([c[..., ts + 1 - a] - c[..., ts + 1 - b] for a, b in HIST], -1)
    return hist, c[..., ts + 1 + H] - c[..., ts + 1]


def environment(ts, left):
    b = _bumps(ts.astype(float), np.linspace(ts[0], ts[-1], 6))[None].repeat(len(left), 0)
    return np.concatenate([b, b * left[:, None, None]], -1)


def _folds(nt, rng):
    return np.array_split(rng.permutation(nt), FOLDS)


def share(x, ts, left, rng, virtual=False):
    """S_proj on the rows ts, every model cross-fitted over trials; x (neurons, trials, bins)."""
    n, nt, _ = x.shape
    env = environment(ts, left)
    nr = len(ts)
    hist, target = windows(x, ts)
    folds = _folds(nt, rng)
    pred = np.empty_like(target)
    envf = env.reshape(nt * nr, -1)
    for i in range(n):
        xi = np.column_stack([own(hist[i]).reshape(nt * nr, -1), envf])
        yi = target[i].ravel()
        for f in folds:
            train = np.ones(nt, bool)
            train[f] = False
            rows = np.repeat(train, nr)
            p, _ = poisson_glm(xi, yi, rows)
            pred[i][f] = p.reshape(nt, nr)[f]
    if virtual:
        perms = []
        for _ in range(n):
            p = np.arange(nt)
            for side in (True, False):
                idx = np.flatnonzero(left == side)
                p[idx] = rng.permutation(idx)
            perms.append(p)
        x = np.stack([x[i][p] for i, p in enumerate(perms)])
        pred = np.stack([pred[i][p] for i, p in enumerate(perms)])
    phi_hist, phi = windows(x.sum(0), ts)
    g0 = pred.sum(0)
    xg = np.column_stack([group(phi_hist).reshape(nt * nr, -1), envf])
    y, y0 = phi.ravel(), g0.ravel()
    e_g, e_0 = np.empty(nt), np.empty(nt)
    for f in folds:
        train = np.ones(nt, bool)
        train[f] = False
        rows = np.repeat(train, nr)
        pg = ridge(xg, y, rows).reshape(nt, nr)
        p0 = ridge(xg, y0, rows).reshape(nt, nr)
        e_g[f] = ((phi[f] - pg[f]) ** 2).sum(1)
        e_0[f] = ((phi[f] - p0[f]) ** 2).sum(1)
    s = 1 - e_g.sum() / e_0.sum()
    idx = rng.integers(0, nt, (N_BOOT, nt))
    boot = 1 - e_g[idx].sum(1) / e_0[idx].sum(1)
    lo, hi = np.quantile(boot, [0.05, 0.95])
    return {"S": float(s), "lo": float(lo), "hi": float(hi)}


def analyse(x, silenced, left, rng):
    """Both conditions (control subsampled to the silenced count, DRAWS times) in both windows."""
    out = {}
    stim = np.flatnonzero(silenced)
    ctrl = np.flatnonzero(~silenced)
    for w, ts in WINDOWS.items():
        res = {"silenced": [], "control": [], "silenced_virtual": [], "control_virtual": []}
        for d in range(DRAWS):
            sub = rng.choice(ctrl, len(stim), replace=False)
            for cond, tr in (("silenced", stim), ("control", sub)):
                if cond == "silenced" and d > 0:
                    continue
                res[cond].append(share(x[:, tr], ts, left[tr], rng))
                res[cond + "_virtual"].append(share(x[:, tr], ts, left[tr], rng, virtual=True))
        out[w] = {k: {"S": float(np.mean([r["S"] for r in v])), "lo": v[0]["lo"], "hi": v[0]["hi"],
                      "draws": [r["S"] for r in v]} for k, v in res.items()}
    pre = slice(CUE - 30, CUE)
    post = slice(CUE, CUE + 30)
    tot = x.sum(0).astype(float)
    rise = lambda tr: float(tot[tr][:, post].mean() - tot[tr][:, pre].mean())  # noqa: E731
    out["rise"] = {"silenced": rise(stim), "control": rise(ctrl)}
    return out


# ---------------------------------------------------------------- part A: simulated networks

def simulate(cond, seed, n=40, trials=80):
    rng = np.random.default_rng([SEED, SIM_CONDITIONS.index(cond), seed])
    nb = 300
    silenced = np.arange(trials) >= trials // 2
    left = rng.random(trials) < 0.5
    base = np.log(rng.uniform(2, 15, n) * 0.01)
    lat = rng.uniform(5, 15, n)
    t = np.arange(nb) - CUE
    kern = np.where(t[None] > 0, (t[None] / lat[:, None]) * np.exp(1 - t[None] / lat[:, None]), 0.0)
    side = rng.normal(0, 0.6, n)
    cue = kern[:, None] * (0.8 + side[:, None, None] * (2 * left[None, :, None] - 1))
    light = (np.arange(nb) >= CUE - 60) & (np.arange(nb) < CUE + 60)
    coupled = cond == "coupled"
    if coupled:
        cue = cue * ~(silenced[None, :, None] & light[None, None, :])
    w = np.where(rng.random((n, n)) < 0.2, 0.12 + 0.12 * rng.normal(size=(n, n)), 0.0) * coupled
    np.fill_diagonal(w, 0.0)
    load = rng.normal(0.4, 0.15, n) * (cond == "hidden")
    z = np.zeros((trials, nb))
    for k in range(1, nb):
        z[:, k] = z[:, k - 1] * (1 - 0.01 / 0.3) + np.sqrt(2 * 0.01 / 0.3) * rng.normal(size=trials)
    z = z * ~(silenced[:, None] & light[None, :])
    x = np.zeros((n, trials, nb), dtype=np.int16)
    for k in range(nb):
        eta = base[:, None] + cue[:, :, k] + load[:, None] * z[None, :, k]
        if k >= 1:
            eta = eta - 2.0 * x[:, :, k - 1]
            if coupled:
                eta = eta + w @ (x[:, :, k - 1] + (x[:, :, k - 2] if k >= 2 else 0))
        x[:, :, k] = rng.poisson(np.exp(np.minimum(eta, 0.0)))
    return x, silenced, left


def _sim_run(job):
    cond, seed = job
    x, silenced, left = simulate(cond, seed)
    rng = np.random.default_rng([SEED, 1, seed])
    out = {}
    for w, ts in WINDOWS.items():
        out[w] = {c: share(x[:, tr], ts, left[tr], rng)
                  for c, tr in (("silenced", np.flatnonzero(silenced)), ("control", np.flatnonzero(~silenced)))}
    return {"cond": cond, "seed": seed, "rate": float(x.mean() / 0.01), **out}


# ---------------------------------------------------------------- part B: real sessions

def eligible():
    from experiments import inagaki_data as idata
    out = []
    for r in idata.sessions():
        s = idata.load(r["Session_name"])
        if s["x"].shape[0] >= MIN_UNITS and s["silenced"].sum() >= MIN_SILENCED:
            out.append(r["Session_name"])
    return out


def _session_run(name):
    from experiments import inagaki_data as idata
    s = idata.load(name)
    res = analyse(s["x"], s["silenced"], s["left"], np.random.default_rng([SEED, 2, len(name), s["x"].shape[0]]))
    return {"session": name, "units": int(s["x"].shape[0]), "silenced": int(s["silenced"].sum()),
            "control": int((~s["silenced"]).sum()), **res}


# ---------------------------------------------------------------- analysis

def _sign_p(diffs):
    """One-sided sign test: P(at least this many negatives | no effect)."""
    d = [v for v in diffs if v != 0]
    k = sum(v < 0 for v in d)
    n = len(d)
    return sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else 1.0


def effects(r):
    d_light = r["light"]["silenced"]["S"] - r["light"]["control"]["S"]
    d_base = r["baseline"]["silenced"]["S"] - r["baseline"]["control"]["S"]
    return d_light, d_base


def predictions(res):
    sim = {c: [r for r in res["sim"] if r["cond"] == c] for c in SIM_CONDITIONS}
    ns = len(SIM_SEEDS)
    a1 = sum(all(abs(r[w][c]["S"]) <= 0.02 for w in ("light",) for c in ("silenced", "control"))
             for r in sim["independent"])
    a2 = sum(r["light"]["silenced"]["S"] - r["light"]["control"]["S"] <= -0.1 for r in sim["hidden"])
    a3 = sum(abs(r["light"]["silenced"]["S"] - r["light"]["control"]["S"]) <= 0.03 for r in sim["coupled"])
    real = res["real"]
    m = len(real)
    r0 = sum(r["rise"]["silenced"] < r["rise"]["control"] for r in real)
    cells = [r[w][k]["S"] for r in real for w in WINDOWS for k in ("silenced_virtual", "control_virtual")]
    r1 = sum(abs(v) <= 0.03 for v in cells)
    dl, db = zip(*[effects(r) for r in real]) if real else ((), ())
    med_b = float(np.median(db)) if real else np.nan
    did = [a - b for a, b in zip(dl, db)]
    med_did = float(np.median(did)) if real else np.nan
    p = _sign_p(did)
    s_sil = float(np.median([r["light"]["silenced"]["S"] for r in real])) if real else np.nan
    v_sil = float(np.median([r["light"]["silenced_virtual"]["S"] for r in real])) if real else np.nan
    return [
        ("A1", "simulated independent: |S| <= 0.02 in both conditions, >= 4/5 seeds", a1 >= 4, f"{a1}/{ns}"),
        ("A2", "simulated hidden input switched off: S silenced - control <= -0.1, >= 4/5 seeds", a2 >= 4, f"{a2}/{ns}"),
        ("A3", "simulated local coupling, cue input removed: |S silenced - control| <= 0.03, >= 4/5", a3 >= 4,
         f"{a3}/{ns}"),
        ("R0", "silencing works: the cue-triggered rise is smaller in silenced trials in >= 75% of sessions",
         r0 >= 0.75 * m, f"{r0}/{m}"),
        ("R1", "virtual groups clean: |S_virtual| <= 0.03 in >= 80% of cells", r1 >= 0.8 * len(cells),
         f"{r1}/{len(cells)}"),
        ("R2", "before the light nothing differs: |median S(baseline) silenced - control| <= 0.02",
         abs(med_b) <= 0.02, f"median {med_b:+.3f}"),
        ("R3", "ALM's coordination depends on the midbrain input: median difference-in-differences < 0, "
               "sign test p < 0.05", med_did < 0 and p < 0.05, f"median {med_did:+.3f}, p = {p:.3f}"),
        ("R4", "coordination survives the input's removal: median S(light, silenced) > virtual + 0.02",
         s_sil > v_sil + 0.02, f"S {s_sil:+.3f}, virtual {v_sil:+.3f}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 64: does a brain area's interaction share survive switching off a major input?", "",
             "  Simulated networks (light window S, control -> silenced; baseline control -> silenced):"]
    for r in res["sim"]:
        lines.append(f"  {r['cond']:12} seed {r['seed']}  rate {r['rate']:5.1f} Hz  light "
                     f"{r['light']['control']['S']:+.3f} -> {r['light']['silenced']['S']:+.3f}   baseline "
                     f"{r['baseline']['control']['S']:+.3f} -> {r['baseline']['silenced']['S']:+.3f}")
    lines += ["", "  ALM sessions (S control -> silenced; virtual; cue-triggered rise control / silenced):"]
    for r in res["real"]:
        dl, db = effects(r)
        lines.append(f"  {r['session']:13} units {r['units']:2d} trials {r['silenced']:2d}/{r['control']:3d}  light "
                     f"{r['light']['control']['S']:+.3f} -> {r['light']['silenced']['S']:+.3f} (d {dl:+.3f})  "
                     f"baseline {r['baseline']['control']['S']:+.3f} -> {r['baseline']['silenced']['S']:+.3f} "
                     f"(d {db:+.3f})  virtual {r['light']['silenced_virtual']['S']:+.3f}  rise "
                     f"{r['rise']['control']:+.2f} / {r['rise']['silenced']:+.2f}")
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
    sim_jobs = [(c, s) for c in SIM_CONDITIONS for s in SIM_SEEDS]
    names = eligible()
    if quick:
        sim_jobs, names = sim_jobs[::5], names[:1]
    with ProcessPoolExecutor(max_workers=6) as pool:
        sim = list(pool.map(_sim_run, sim_jobs))
        real = list(pool.map(_session_run, names))
    res = {"sim": sim, "real": real}
    (out_dir / "e64_input_silencing_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e64_figures import make_figures
    make_figures(res, out_dir / "e64_figures", text)
    return text


if __name__ == "__main__":
    main()
