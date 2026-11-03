"""Experiment 65: does ALM's coordination depend on its midbrain input? (e64 redesigned on simulations)

e64 scored 2/8 and its real-data numbers are void. Its own simulations showed why: its light window straddled the
go cue, whose time-locked transient dominates a share estimated from about 40 trials. Independent networks scored
+0.04 to +0.19 there instead of 0, and removing the cue input from a coupled network moved the share more than
removing a hidden input. Its Poisson twins also overfitted, giving -0.1 in quiet windows, and occasionally blew up
on held-out trials.

The redesign was developed on simulations only (seeds 10-19), before any real share in the new window was
computed:
  * Window: the pre-cue part of the light, from 0.6 s before the cue to the cue (history and 50 ms target
    inside), a steady delay state. Baseline: the delay before the light, as e64.
  * Twin: per-neuron ridge regression (penalty 10, standardized) on its own counts in the last 5, 5-10, 10-20
    bins and its cross-fitted trial-average for the instructed side at that moment (PSTH, from the other folds).
    Group law: the same ridge on the ALM total's three windows and the summed PSTH. Linear twins cannot blow up.
  * Everything else as e64: 5-fold cross-fitting over trials, control trials subsampled to the silenced count
    (5 draws averaged), virtual groups permuted within side, trial bootstrap.
On development seeds this gave independent and coupled networks -0.02 to -0.01, and a hidden input 0.30 that
fell to -0.03 when switched off. Realistic coupling is not detectable at this sample size.

Data: e64's 21 sessions. Simulations: e64's networks, fresh seeds 20-29.

Predictions, fixed before running:
  B1  Independent networks: |S| <= 0.03 in both conditions (pre-cue) in >= 8/10 seeds.
  B2  A hidden input switched off is seen: S silenced - control (pre-cue) <= -0.1 in >= 8/10 seeds.
  B3  Local coupling is not mistaken for input: coupled, |S silenced - control| (pre-cue) <= 0.03 in >= 8/10.
  R1  Virtual groups are clean: |S_virtual| <= 0.03 in >= 80% of session x condition x window cells.
  R2  Before the light nothing differs: |median S(baseline) silenced - control| <= 0.02.
  R3  There is coordination to remove: median S(pre-cue, control) >= 0.02.
  R4  ALM's coordination depends on the midbrain input: median of [S(pre-cue) silenced - control] -
      [S(baseline) silenced - control] < 0, one-sided sign test p < 0.05.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e61_neural_share import ridge
from experiments.e64_input_silencing import _folds, _sign_p, eligible, simulate, windows

SEED = 65
WINDOWS = {"precue": np.arange(160, 195, 2), "baseline": np.arange(100, 135, 2)}
LAM = 10.0
DRAWS = 5
N_BOOT = 400
SIM_SEEDS = tuple(range(20, 30))
SIM_CONDITIONS = ("independent", "coupled", "hidden")


def _perm(left, nt, rng):
    p = np.arange(nt)
    for side in (True, False):
        idx = np.flatnonzero(left == side)
        p[idx] = rng.permutation(idx)
    return p


def share(x, ts, left, rng, virtual=False):
    """Cross-fitted S_proj with linear twins and a PSTH environment."""
    n, nt, _ = x.shape
    nr = len(ts)
    hist, target = windows(x, ts)
    folds = _folds(nt, rng)
    psth = np.empty_like(target)
    for f in folds:
        tr = np.ones(nt, bool)
        tr[f] = False
        for side in (True, False):
            sel = np.flatnonzero(~tr & (left == side))
            psth[:, sel] = target[:, tr & (left == side)].mean(1)[:, None]
    pred = np.empty_like(target)
    for i in range(n):
        xi = np.column_stack([hist[i].reshape(nt * nr, -1), psth[i].ravel()])
        yi = target[i].ravel()
        for f in folds:
            tr = np.ones(nt, bool)
            tr[f] = False
            pred[i][f] = ridge(xi, yi, np.repeat(tr, nr), LAM).reshape(nt, nr)[f]
    if virtual:
        perms = [_perm(left, nt, rng) for _ in range(n)]
        x = np.stack([x[i][p] for i, p in enumerate(perms)])
        pred = np.stack([pred[i][p] for i, p in enumerate(perms)])
        psth = np.stack([psth[i][p] for i, p in enumerate(perms)])
    phi_hist, phi = windows(x.sum(0), ts)
    xg = np.column_stack([phi_hist.reshape(nt * nr, -1), psth.sum(0).ravel()])
    y, y0 = phi.ravel(), pred.sum(0).ravel()
    e_g, e_0 = np.empty(nt), np.empty(nt)
    for f in folds:
        tr = np.ones(nt, bool)
        tr[f] = False
        rows = np.repeat(tr, nr)
        e_g[f] = ((phi[f] - ridge(xg, y, rows, LAM).reshape(nt, nr)[f]) ** 2).sum(1)
        e_0[f] = ((phi[f] - ridge(xg, y0, rows, LAM).reshape(nt, nr)[f]) ** 2).sum(1)
    s = 1 - e_g.sum() / e_0.sum()
    idx = rng.integers(0, nt, (N_BOOT, nt))
    boot = 1 - e_g[idx].sum(1) / e_0[idx].sum(1)
    lo, hi = np.quantile(boot, [0.05, 0.95])
    return {"S": float(s), "lo": float(lo), "hi": float(hi)}


def analyse(x, silenced, left, rng):
    out = {}
    stim, ctrl = np.flatnonzero(silenced), np.flatnonzero(~silenced)
    for w, ts in WINDOWS.items():
        res = {"silenced": [], "control": [], "silenced_virtual": [], "control_virtual": []}
        for d in range(DRAWS):
            sub = rng.choice(ctrl, len(stim), replace=False)
            for cond, tr in (("silenced", stim), ("control", sub)):
                if cond == "silenced" and d > 0:
                    continue
                res[cond].append(share(x[:, tr], ts, left[tr], rng))
                res[cond + "_virtual"].append(share(x[:, tr], ts, left[tr], rng, virtual=True))
        out[w] = {k: {"S": float(np.mean([r["S"] for r in v])), "lo": v[0]["lo"], "hi": v[0]["hi"]}
                  for k, v in res.items()}
    tot = x.sum(0).astype(float)
    ts = WINDOWS["precue"]
    out["rate"] = {"silenced": float(tot[stim][:, ts].mean() / 0.01), "control": float(tot[ctrl][:, ts].mean() / 0.01)}
    return out


def _sim_run(job):
    cond, seed = job
    x, silenced, left = simulate(cond, seed)
    rng = np.random.default_rng([SEED, 1, seed])
    out = {w: {c: share(x[:, tr], ts, left[tr], rng)
               for c, tr in (("silenced", np.flatnonzero(silenced)), ("control", np.flatnonzero(~silenced)))}
           for w, ts in WINDOWS.items()}
    return {"cond": cond, "seed": seed, **out}


def _session_run(name):
    from experiments import inagaki_data as idata
    s = idata.load(name)
    res = analyse(s["x"], s["silenced"], s["left"], np.random.default_rng([SEED, 2, len(name), s["x"].shape[0]]))
    return {"session": name, "units": int(s["x"].shape[0]), "silenced": int(s["silenced"].sum()),
            "control": int((~s["silenced"]).sum()), **res}


def effects(r):
    return (r["precue"]["silenced"]["S"] - r["precue"]["control"]["S"],
            r["baseline"]["silenced"]["S"] - r["baseline"]["control"]["S"])


def predictions(res):
    sim = {c: [r for r in res["sim"] if r["cond"] == c] for c in SIM_CONDITIONS}
    ns = len(SIM_SEEDS)
    d = lambda r: r["precue"]["silenced"]["S"] - r["precue"]["control"]["S"]  # noqa: E731
    b1 = sum(all(abs(r["precue"][c]["S"]) <= 0.03 for c in ("silenced", "control")) for r in sim["independent"])
    b2 = sum(d(r) <= -0.1 for r in sim["hidden"])
    b3 = sum(abs(d(r)) <= 0.03 for r in sim["coupled"])
    real = res["real"]
    cells = [r[w][k]["S"] for r in real for w in WINDOWS for k in ("silenced_virtual", "control_virtual")]
    r1 = sum(abs(v) <= 0.03 for v in cells)
    dp, db = zip(*[effects(r) for r in real]) if real else ((), ())
    med_b = float(np.median(db)) if real else np.nan
    med_c = float(np.median([r["precue"]["control"]["S"] for r in real])) if real else np.nan
    did = [a - b for a, b in zip(dp, db)]
    med_did = float(np.median(did)) if real else np.nan
    p = _sign_p(did)
    return [
        ("B1", "simulated independent: |S| <= 0.03 in both conditions, >= 8/10 seeds", b1 >= 8, f"{b1}/{ns}"),
        ("B2", "simulated hidden input switched off: S silenced - control <= -0.1, >= 8/10 seeds", b2 >= 8,
         f"{b2}/{ns}"),
        ("B3", "simulated local coupling: |S silenced - control| <= 0.03, >= 8/10 seeds", b3 >= 8, f"{b3}/{ns}"),
        ("R1", "virtual groups clean: |S_virtual| <= 0.03 in >= 80% of cells", r1 >= 0.8 * len(cells),
         f"{r1}/{len(cells)}"),
        ("R2", "before the light nothing differs: |median S(baseline) silenced - control| <= 0.02",
         abs(med_b) <= 0.02, f"median {med_b:+.3f}"),
        ("R3", "there is coordination to remove: median S(pre-cue, control) >= 0.02", med_c >= 0.02,
         f"median {med_c:+.3f}"),
        ("R4", "ALM's coordination depends on the midbrain input: median difference-in-differences < 0, "
               "sign test p < 0.05", med_did < 0 and p < 0.05, f"median {med_did:+.3f}, p = {p:.3f}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 65: does ALM's coordination depend on its midbrain input? (pre-cue window)", "",
             "  Simulated networks (pre-cue S control -> silenced; baseline control -> silenced):"]
    for r in res["sim"]:
        lines.append(f"  {r['cond']:12} seed {r['seed']}  pre-cue {r['precue']['control']['S']:+.3f} -> "
                     f"{r['precue']['silenced']['S']:+.3f}   baseline {r['baseline']['control']['S']:+.3f} -> "
                     f"{r['baseline']['silenced']['S']:+.3f}")
    lines += ["", "  ALM sessions (S control -> silenced; virtual control / silenced; pre-cue rate control / silenced):"]
    for r in res["real"]:
        dp, db = effects(r)
        lines.append(f"  {r['session']:13} units {r['units']:2d} trials {r['silenced']:2d}/{r['control']:3d}  pre-cue "
                     f"{r['precue']['control']['S']:+.3f} -> {r['precue']['silenced']['S']:+.3f} (d {dp:+.3f})  "
                     f"baseline {r['baseline']['control']['S']:+.3f} -> {r['baseline']['silenced']['S']:+.3f} "
                     f"(d {db:+.3f})  virtual {r['precue']['control_virtual']['S']:+.3f} / "
                     f"{r['precue']['silenced_virtual']['S']:+.3f}  rate {r['rate']['control']:.0f} / "
                     f"{r['rate']['silenced']:.0f} Hz")
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
        sim_jobs, names = sim_jobs[::10], names[:1]
    with ProcessPoolExecutor(max_workers=6) as pool:
        sim = list(pool.map(_sim_run, sim_jobs))
        real = list(pool.map(_session_run, names))
    res = {"sim": sim, "real": real}
    (out_dir / "e65_input_silencing_precue_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e65_figures import make_figures
    make_figures(res, out_dir / "e65_figures", text)
    return text


if __name__ == "__main__":
    main()
