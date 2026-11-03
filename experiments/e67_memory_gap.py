"""Experiment 67: one equation for the interaction share across domains (the memory gap).

Across e9-e66 the projected share was high where the group's past reveals something no member's own past reveals
(fish schools; neurons with a hidden drive) and zero where every member already carries the shared signal (markets).
For linear twins and laws this can be made exact. A member's twin weights its own past by the member's own memory
(autocovariance o(k), averaged over members); the group law weights the group's past by the group's memory g(k).
For exchangeable members the summed twin forecast is the member weights applied to the group's past, so

    S_pred = 1 - MSE*(g) / MSE(beta(o); g),

the loss from forecasting the group with its members' memory instead of its own. Everything is fixed by o(k), g(k)
and the feature windows. With one lag it reduces to the memory-gap equation

    S_1 = (rho_g - rho_m)^2 / (1 - 2 rho_m rho_g + rho_m^2),   rho_m = o(1)/o(0), rho_g = g(1)/g(0).

S is zero exactly when the group's memory has the shape of its members' memory, whatever the members' correlation.
A shared signal raises S when it is persistent and each member reveals it poorly (low per-member share, many
members). The equation depends on second-order statistics only: a common input and all-to-all coupling that give
the same o and g give the same S. It predicts S; it cannot by itself say which mechanism made it.

Measured S: the cross-fitted linear projected share (e65's construction without environment) on the same
windows, on data from which recorded causes were removed beforehand (cross-fitted condition means for trials).
S_pred and S_1 use autocovariances estimated from the same data.

Development: simulations only (seeds 100+). No real-data S_pred was computed before registration. Real measured
values seen before: e66 industries (linear HAR S = -0.005); e62 brain areas (Poisson twins, E0).

Part A, simulations (fresh seeds):
  factor     x_i = a c + eta_i; c AR(1) (rho_c 0.9, 0.98); eta_i AR(1) 0.3; per-member shared share
             r = a^2 var c / var x in {0.003, 0.01, 0.03, 0.1, 0.3, 0.7}; N in {10, 50, 200}; 20 000 steps.
  meanfield  x_i(t+1) = 0.3 x_i(t) + kappa mean_j x_j(t) + noise; kappa in {0.1, 0.3, 0.5, 0.65}; N as above.
  local      each member driven by 5 random others with total weight kappa in {0.3, 0.6}; N in {50, 200}.
  garch      e66's simulated markets (independent, correlated, spillover, factor), 2 seeds, HAR windows, |r|.
  spiking    e62's networks (independent, coupled, hidden), 2 seeds, e62's windows, condition means removed.
Part B, real data: e66's 49 industries (HAR windows); e62's 221 brain areas (e62's windows, rows and neurons;
condition = sign of contrast difference; cross-fitted condition means removed per neuron and bin).

Predictions, fixed before running:
  P1  Linear systems (factor, meanfield, local): |S - S_pred| <= 0.02 + 0.1 S_pred in >= 90% of cells.
  P2  Mechanism-blind: the meanfield and local cells meet P1's tolerance as often (>= 90%) as the factor cells.
  P3  The pooling peak: in each factor series (N, rho_c) measured S is highest at an interior r (not the
      smallest or largest), in >= 5 of 6 series; and the r of the peak is no larger at N = 200 than at N = 10 for
      both rho_c.
  P4  The one-lag equation ranks the linear cells: Spearman(S_1, S) >= 0.8.
  P5  Non-Gaussian simulations: |S - S_pred| <= 0.02 in >= 80% of the garch and spiking cells.
  R1  Industries: |S - S_pred| <= 0.01.
  R2  Brain areas: Spearman(S_pred, S) >= 0.7 and median |S - S_pred| <= 0.01.
  R3  The equation explains e62's Poisson-twin result: Spearman(S_pred, e62 S_proj(E0)) >= 0.5.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SEED = 67
LAM = 1.0
FOLDS = 5
SERIES_WINDOWS = ((1, 2), (2, 6), (6, 21))   # lags [a, b): x(t-a) .. x(t-b+1); target x(t)
HAR_WINDOWS = ((1, 2), (1, 6), (1, 23))
R_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 0.7)
N_GRID = (10, 50, 200)
RHO_C = (0.9, 0.98)
KAPPA_MF = (0.1, 0.3, 0.5, 0.65)
KAPPA_LOCAL = (0.3, 0.6)
T_SERIES = 20000


# ---------------------------------------------------------------- the equation

def _toeplitz(c, offsets):
    d = np.abs(offsets[:, None] - offsets[None, :])
    return c[d]


def design(windows, target):
    """Weights over time offsets for features (rows) and target. windows: lag ranges [a, b) before t (series)
    or (start, stop) offsets; target: list of offsets. Returns (offsets, F, y)."""
    feats = [np.arange(-b + 1, -a + 1) for a, b in windows]
    offs = np.arange(min(min(f.min() for f in feats), min(target)), max(max(f.max() for f in feats), max(target)) + 1)
    f = np.array([np.isin(offs, w).astype(float) for w in feats])
    y = np.isin(offs, target).astype(float)
    return offs, f, y


def s_pred(o, g, offs, f, y):
    """Population projected share of the group from member autocovariance o and group autocovariance g."""
    mo, mg = _toeplitz(o, offs - offs.min()), _toeplitz(g, offs - offs.min())
    beta = np.linalg.solve(f @ mo @ f.T, f @ mo @ y)
    a, b, v = f @ mg @ f.T, f @ mg @ y, y @ mg @ y
    opt = v - b @ np.linalg.solve(a, b)
    tw = v - 2 * beta @ b + beta @ a @ beta
    return float(1 - opt / tw)


def s_one(o, g):
    rm, rg = o[1] / o[0], g[1] / g[0]
    return float((rg - rm) ** 2 / (1 - 2 * rm * rg + rm ** 2))


def autocov(x, maxlag):
    """x (members, segments, T), already centred. Returns member-mean o(k) and group g(k), k = 0..maxlag."""
    phi = x.sum(0)
    t = x.shape[-1]
    o = np.array([(x[..., : t - k] * x[..., k:]).mean() for k in range(maxlag + 1)])
    g = np.array([(phi[..., : t - k] * phi[..., k:]).mean() for k in range(maxlag + 1)])
    return o, g


# ---------------------------------------------------------------- measured linear share

def _ridge(x, y, train, lam=LAM):
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-12
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    a = z[train].T @ z[train] + lam * np.diag([0.0] + [1.0] * (z.shape[1] - 1))
    return z @ np.linalg.solve(a, z[train].T @ y[train])


def features(x, offs, f, y, rows):
    """x (..., T); rows: time indices t. Returns features (..., rows, k) and target (..., rows)."""
    idx = rows[:, None] + offs[None, :]
    seg = x[..., idx]
    return seg @ f.T, seg @ y


def measured(x, offs, f, y, rows, fold_of_segment):
    """Cross-fitted linear S_proj. x (members, segments, T); folds over segments."""
    n, ns = x.shape[:2]
    hist, target = features(x, offs, f, y, rows)
    nr = len(rows)
    fold = np.repeat(fold_of_segment, nr)
    pred = np.empty((n, ns * nr))
    for i in range(n):
        xi, yi = hist[i].reshape(ns * nr, -1), target[i].ravel()
        for k in np.unique(fold):
            pred[i, fold == k] = _ridge(xi, yi, fold != k)[fold == k]
    gh, gt = features(x.sum(0), offs, f, y, rows)
    xg, yg, y0 = gh.reshape(ns * nr, -1), gt.ravel(), pred.sum(0)
    e_g, e_0 = 0.0, 0.0
    for k in np.unique(fold):
        te = fold == k
        e_g += ((yg[te] - _ridge(xg, yg, ~te)[te]) ** 2).sum()
        e_0 += ((yg[te] - _ridge(xg, y0, ~te)[te]) ** 2).sum()
    return float(1 - e_g / e_0)


def evaluate(x, offs, f, y, rows, fold_of_segment):
    maxlag = int(offs.max() - offs.min())
    o, g = autocov(x - x.mean(axis=(1, 2), keepdims=True) if x.shape[1] == 1 else x, maxlag)
    return {"S": measured(x, offs, f, y, rows, fold_of_segment), "S_pred": s_pred(o, g, offs, f, y),
            "S_1": s_one(o, g), "rho_m": float(o[1] / o[0]), "rho_g": float(g[1] / g[0]),
            "corr": float((g[0] - len(x) * o[0]) / (len(x) * (len(x) - 1) * o[0]))}


def series_design(windows=SERIES_WINDOWS):
    offs, f, y = design([(a, b) for a, b in windows], [0])
    return offs, f, y


def evaluate_series(x, windows=SERIES_WINDOWS):
    """x (members, T): one long series, 3 folds of interleaved blocks."""
    offs, f, y = series_design(windows)
    x = (x - x.mean(1, keepdims=True))[:, None, :]
    rows = np.arange(-offs.min(), x.shape[-1] - offs.max())
    block = 1000
    nb = len(rows) // block
    rows = rows[: nb * block]
    xs = np.stack([x[:, 0, rows[k * block] + offs.min(): rows[(k + 1) * block - 1] + offs.max() + 1] for k in range(nb)],
                  1)
    r = np.arange(-offs.min(), xs.shape[-1] - offs.max())
    return evaluate(xs, offs, f, y, r, np.arange(nb) % 3)


# ---------------------------------------------------------------- simulations

def sim_factor(n, r, rho_c, seed, t=T_SERIES):
    rng = np.random.default_rng([SEED, 1, seed, n, int(r * 1e4), int(rho_c * 100)])
    c = np.zeros(t)
    e = rng.standard_normal(t) * np.sqrt(1 - rho_c ** 2)
    for k in range(1, t):
        c[k] = rho_c * c[k - 1] + e[k]
    eta = np.zeros((n, t))
    z = rng.standard_normal((n, t)) * np.sqrt(1 - 0.09)
    for k in range(1, t):
        eta[:, k] = 0.3 * eta[:, k - 1] + z[:, k]
    return np.sqrt(r) * c[None] + np.sqrt(1 - r) * eta


def sim_meanfield(n, kappa, seed, t=T_SERIES):
    rng = np.random.default_rng([SEED, 2, seed, n, int(kappa * 100)])
    x = np.zeros((n, t))
    z = rng.standard_normal((n, t))
    for k in range(1, t):
        x[:, k] = 0.3 * x[:, k - 1] + kappa * x[:, k - 1].mean() + z[:, k]
    return x


def sim_local(n, kappa, seed, t=T_SERIES):
    rng = np.random.default_rng([SEED, 3, seed, n, int(kappa * 100)])
    w = np.zeros((n, n))
    for i in range(n):
        w[i, rng.choice(np.delete(np.arange(n), i), 5, replace=False)] = kappa / 5
    x = np.zeros((n, t))
    z = rng.standard_normal((n, t))
    for k in range(1, t):
        x[:, k] = 0.3 * x[:, k - 1] + w @ x[:, k - 1] + z[:, k]
    return x


def _job_linear(job):
    kind, n, p, rho, seed = job
    if kind == "factor":
        x = sim_factor(n, p, rho, seed)
    elif kind == "meanfield":
        x = sim_meanfield(n, p, seed)
    else:
        x = sim_local(n, p, seed)
    return {"kind": kind, "n": n, "p": p, "rho_c": rho, "seed": seed, **evaluate_series(x)}


def _job_garch(job):
    from experiments import e66_market_volatility as e66
    cond, seed = job
    x = e66.simulate(cond, seed + 200).T
    return {"kind": "garch", "cond": cond, "seed": seed, **evaluate_series(x, HAR_WINDOWS)}


# ---------------------------------------------------------------- trials (spiking, brain areas)

def trial_design():
    from experiments.e62_neural_share_repaired import H, HIST, TS
    offs, f, y = design([(a, b) for a, b in HIST], list(range(1, H + 1)))
    return offs, f, y, TS


def residualize(x, cond, fold):
    """Remove cross-fitted per-neuron, per-bin means within each trial condition."""
    out = np.empty(x.shape, float)
    for k in np.unique(fold):
        te = fold == k
        for c in np.unique(cond):
            tr = ~te & (cond == c)
            sel = te & (cond == c)
            m = x[:, tr].mean(1, keepdims=True) if tr.any() else x[:, ~te].mean(1, keepdims=True)
            out[:, sel] = x[:, sel] - m
    return out


def evaluate_trials(x, cl, cr, seed):
    offs, f, y, ts = trial_design()
    rng = np.random.default_rng([SEED, 4, seed])
    nt = x.shape[1]
    fold = rng.permutation(np.arange(nt) % FOLDS)
    cond = np.sign(cl - cr).astype(int)
    xr = residualize(x.astype(float), cond, fold)
    return evaluate(xr, offs, f, y, ts, fold)


def _job_spiking(job):
    from experiments.e62_neural_share_repaired import simulate
    cond, seed = job
    x, cl, cr, _ = simulate(cond, seed + 200)
    return {"kind": "spiking", "cond": cond, "seed": seed, **evaluate_trials(x, cl, cr, seed)}


def _job_area(s):
    from experiments import steinmetz_data as sd
    from experiments.e62_neural_share_repaired import MIN_NEURONS, MIN_RATE
    d = sd.session(s)
    td = sd.trial_data(s)
    spk, names = d["spks"], d["brain_area"]
    rates = spk.mean((1, 2)) / 0.01
    out = []
    for area in np.unique(names):
        sel = (names == area) & (rates >= MIN_RATE)
        if area == "root" or sel.sum() < MIN_NEURONS:
            continue
        r = evaluate_trials(spk[sel], td["cl"], td["cr"], s * 1000 + int(sel.sum()))
        out.append({"session": s, "area": str(area), "n": int(sel.sum()), **r})
    return out


def _job_industries(_):
    from experiments import finance_data as fd
    _, x, _ = fd.industries()
    return {"kind": "industries", **evaluate_series(np.abs(x).T, HAR_WINDOWS)}


# ---------------------------------------------------------------- predictions

def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0]) if len(a) > 2 else np.nan


def _tol(r):
    return abs(r["S"] - r["S_pred"]) <= 0.02 + 0.1 * abs(r["S_pred"])


def predictions(res):
    sim = res["sim"]
    lin = [r for r in sim if r["kind"] in ("factor", "meanfield", "local")]
    fac = [r for r in lin if r["kind"] == "factor"]
    mech = [r for r in lin if r["kind"] != "factor"]
    p1 = sum(_tol(r) for r in lin)
    p2f, p2m = sum(_tol(r) for r in fac), sum(_tol(r) for r in mech)
    series, peaks = 0, {}
    for n in N_GRID:
        for rho in RHO_C:
            s = [next(r["S"] for r in fac if r["n"] == n and r["rho_c"] == rho and r["p"] == p) for p in R_GRID]
            k = int(np.argmax(s))
            series += 0 < k < len(R_GRID) - 1
            peaks[(n, rho)] = R_GRID[k]
    p3b = all(peaks[(N_GRID[-1], rho)] <= peaks[(N_GRID[0], rho)] for rho in RHO_C)
    p4 = _spearman([r["S_1"] for r in lin], [r["S"] for r in lin])
    ng = [r for r in sim if r["kind"] in ("garch", "spiking")]
    p5 = sum(abs(r["S"] - r["S_pred"]) <= 0.02 for r in ng)
    ind = res.get("industries")
    areas = res.get("areas", [])
    d1 = abs(ind["S"] - ind["S_pred"]) if ind else np.nan
    rho2 = _spearman([a["S_pred"] for a in areas], [a["S"] for a in areas])
    med2 = float(np.median([abs(a["S"] - a["S_pred"]) for a in areas])) if areas else np.nan
    e62 = res.get("e62", {})
    pairs = [(a["S_pred"], e62[f"{a['session']}:{a['area']}"]) for a in areas if f"{a['session']}:{a['area']}" in e62]
    rho3 = _spearman(*zip(*pairs)) if len(pairs) > 2 else np.nan
    return [
        ("P1", "linear systems: |S - S_pred| <= 0.02 + 0.1 S_pred in >= 90% of cells", p1 >= 0.9 * len(lin),
         f"{p1}/{len(lin)}"),
        ("P2", "mechanism-blind: meanfield and local cells meet P1 as often (>= 90%) as factor cells",
         p2m >= 0.9 * len(mech) and p2f >= 0.9 * len(fac), f"factor {p2f}/{len(fac)}, coupled {p2m}/{len(mech)}"),
        ("P3", "pooling peak at an interior r in >= 5/6 series; peak r at N=200 <= at N=10",
         series >= 5 and p3b, f"{series}/6 interior; peaks {peaks}"),
        ("P4", "one-lag equation ranks the linear cells: Spearman(S_1, S) >= 0.8", p4 >= 0.8, f"rho {p4:+.2f}"),
        ("P5", "non-Gaussian simulations: |S - S_pred| <= 0.02 in >= 80%", p5 >= 0.8 * len(ng), f"{p5}/{len(ng)}"),
        ("R1", "industries: |S - S_pred| <= 0.01", d1 <= 0.01,
         f"S {ind['S']:+.4f}, S_pred {ind['S_pred']:+.4f}" if ind else "n/a"),
        ("R2", "brain areas: Spearman(S_pred, S) >= 0.7 and median |S - S_pred| <= 0.01",
         rho2 >= 0.7 and med2 <= 0.01, f"rho {rho2:+.2f}, median |diff| {med2:.4f}, {len(areas)} areas"),
        ("R3", "explains e62's Poisson-twin result: Spearman(S_pred, e62 S(E0)) >= 0.5", rho3 >= 0.5,
         f"rho {rho3:+.2f} ({len(pairs)} areas)"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 67: one equation for the interaction share (the memory gap)", "",
             "  Simulations (S measured, S_pred from memories, S_1 one-lag equation; member / group lag-1 memory):"]
    for r in res["sim"]:
        tag = (f"{r['kind']:9} N {r['n']:3d} p {r['p']:<6} rho_c {r['rho_c']}" if "n" in r
               else f"{r['kind']:9} {r['cond']:12} seed {r['seed']}")
        lines.append(f"  {tag:42} S {r['S']:+.4f}  S_pred {r['S_pred']:+.4f}  S_1 {r['S_1']:+.4f}  "
                     f"rho_m {r['rho_m']:.3f} rho_g {r['rho_g']:.3f} corr {r['corr']:+.4f}")
    if res.get("industries"):
        r = res["industries"]
        lines += ["", f"  Industries: S {r['S']:+.4f}  S_pred {r['S_pred']:+.4f}  S_1 {r['S_1']:+.4f}  "
                      f"rho_m {r['rho_m']:.3f} rho_g {r['rho_g']:.3f} corr {r['corr']:+.4f}"]
    areas = res.get("areas", [])
    if areas:
        s, p = np.array([a["S"] for a in areas]), np.array([a["S_pred"] for a in areas])
        lines += ["", f"  Brain areas ({len(areas)}): median S {np.median(s):+.4f}, median S_pred {np.median(p):+.4f}, "
                      f"median member corr {np.median([a['corr'] for a in areas]):+.4f}"]
        for a in sorted(areas, key=lambda a: -a["S"])[:12]:
            lines.append(f"    {a['area']:6} s{a['session']:02d} n {a['n']:3d}  S {a['S']:+.4f}  S_pred {a['S_pred']:+.4f}  "
                         f"rho_m {a['rho_m']:.3f} rho_g {a['rho_g']:.3f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def _e62_reference():
    p = Path("results/e62_neural_share_repaired_raw.json")
    if not p.exists():
        return {}
    return {f"{r['session']}:{r['area']}": r["E0"]["S"] for r in json.loads(p.read_text())["real"]}


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from concurrent.futures import ProcessPoolExecutor
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    lin = ([("factor", n, r, rho, 0) for n in N_GRID for rho in RHO_C for r in R_GRID]
           + [("meanfield", n, k, None, 0) for n in N_GRID for k in KAPPA_MF]
           + [("local", n, k, None, 0) for n in N_GRID[1:] for k in KAPPA_LOCAL])
    garch = [(c, s) for c in ("independent", "correlated", "spillover", "factor") for s in (0, 1)]
    spiking = [(c, s) for c in ("independent", "coupled", "hidden") for s in (0, 1)]
    sessions = list(range(39))
    if quick:
        lin, garch, spiking, sessions = lin[::9], garch[:1], spiking[:1], [0]
    with ProcessPoolExecutor(max_workers=6) as pool:
        sim = list(pool.map(_job_linear, lin)) + list(pool.map(_job_garch, garch)) + list(pool.map(_job_spiking, spiking))
        ind = next(pool.map(_job_industries, [0]))
        areas = [a for out in pool.map(_job_area, sessions) for a in out]
    res = {"sim": sim, "industries": ind, "areas": areas, "e62": _e62_reference()}
    (out_dir / "e67_memory_gap_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e67_figures import make_figures
    make_figures(res, out_dir / "e67_figures", text)
    return text


if __name__ == "__main__":
    main()
