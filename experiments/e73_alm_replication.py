"""Experiment 73: ALM's small coordination share in an independent laboratory dataset, and the silencing test's power.

e71 confirmed on unseen DANDI 000221 sessions (Inagaki & Chen) that ALM in the 1.5 s before the go cue has a small
share beyond each neuron's side-specific response (median 0.013, 18/19 positive, predicted by the memory gap,
Spearman 0.81). Its silencing test was inconclusive: with ~40 silenced trials per session the control excess was
below the 0.02 the test could resolve. No public dataset was found with >= 100 steady-period silenced trials and
>= 20 ALM units (DANDI 000009: thalamus silencing collapses ALM activity; 000011: ~13 units, ~30 trials per
condition; 000006: no stimulation). Two things are possible instead.

Part A, independent replication. Economo, Viswanathan et al. 2018 (DANDI 000006): different laboratory set-up,
different task (pole location, 1.5 s delay, auditory cue), silicon probes. Sessions with >= 20 units that fire across
the session (19 sessions, 33-75 units, 196-336 trials, loader economo_data). No share has been computed on them.
Pipeline exactly as e71 Part A: good, non-early-lick trials; bins 50-199 (the whole 1.5 s delay, 10 ms); counts minus
cross-fitted side-specific means (e67 residualize, 5 folds); e62 history windows, next 50 ms; measured S = e67's
cross-fitted linear S_proj; S_pred = memory-gap tool; virtual = per-neuron trial permutation within side; beyond
drift = minus each neuron's window mean over the 10 trials either side.

Part B, how many silenced trials would a decisive experiment need? e71's simulated network with a hidden input
whose load (0.085) gives a control excess of about 0.011, the size measured in ALM (calibrated on development seeds
200-211: excess 0.011, DiD median -0.009 at 160 silenced trials). Fresh seeds 300-339, silenced trials per session
40, 80, 160, 320 (equal control trials); the e71 difference-in-differences per simulated session. Power for an
experiment of 21 sessions = share of 2000 bootstrap draws of 21 simulated sessions with one-sided sign test p < 0.05.
False positives from coupled networks (input silenced, coupling intact, no input dependence) at 160 and 320.

Predictions, fixed before running:
  E1  Median measured S >= 0.005 and S > 0 in >= 70% of sessions.
  E2  Spearman(measured S, S_pred) >= 0.7.
  E3  Virtual groups: |S| <= 0.01 in >= 80% of sessions.
  E4  Beyond drift: S > 0 in >= 70% of sessions.
  P1  At 40 silenced trials (e71's situation) power < 0.8.
  P2  Power >= 0.8 is reached at some count <= 320.
  P3  Coupled networks: false-positive rate <= 0.1 at 160 and at 320.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e62_neural_share_repaired import H as H62, HIST as HIST62
from experiments.e71_alm_holdout import A_BINS, A_ROWS, _fold, did, drift_removed, simulate
from experiments.memgap import screen

SEED = 73
MIN_UNITS = 20
LOAD = 0.085
COUNTS = (40, 80, 160, 320)
FP_COUNTS = (160, 320)
SEEDS = tuple(range(300, 340))
SESSIONS = 21
BOOT = 2000


def measure(x, left):
    """e71 Part A on one session: x (units, trials, bins) counts, left (trials,) side."""
    from experiments.e67_memory_gap import evaluate, residualize, trial_design
    x = x.astype(float)
    fold = _fold(x.shape[1], 0)
    offs, f, y, _ = trial_design()

    def share(xx):
        xr = residualize(xx, left.astype(int), fold)[:, :, A_BINS[0]:A_BINS[1]]
        return xr, evaluate(xr, offs, f, y, A_ROWS, fold)["S"]

    xr, real = share(x)
    pred = screen([xr[:, t].T for t in range(xr.shape[1])], HIST62, target=range(1, H62 + 1), baseline="permute")
    rng = np.random.default_rng([SEED, 3, x.shape[0]])
    perm = np.empty_like(x)
    for i in range(x.shape[0]):
        p = np.arange(x.shape[1])
        for side in (True, False):
            idx = np.flatnonzero(left == side)
            p[idx] = rng.permutation(idx)
        perm[i] = x[i, p]
    _, virt = share(perm)
    xd = drift_removed(residualize(x, left.astype(int), fold)[:, :, A_BINS[0]:A_BINS[1]])
    drift = evaluate(xd, offs, f, y, A_ROWS, fold)["S"]
    return {"units": int(x.shape[0]), "trials": int(x.shape[1]), "S": real, "S_pred": pred["S_pred"],
            "floor": pred["virtual"], "virtual": virt, "drift": drift}


def _replicate(name):
    from experiments import economo_data as ed
    s = ed.load(name)
    return {"session": name, **measure(s["x"], s["left"])}


def eligible():
    from experiments import economo_data as ed
    return [n for n in ed.sessions() if ed.load(n)["x"].shape[0] >= MIN_UNITS]


def _sim(job):
    cond, seed, sil = job
    x, silenced, left = simulate(cond, seed, trials=2 * sil, load_mean=LOAD)
    return {"cond": cond, "seed": seed, "silenced_trials": sil, **did(x, silenced, left, seed)}


def _sign_p(k, n):
    from scipy.stats import binomtest
    return float(binomtest(int(k), int(n), 0.5, alternative="greater").pvalue)


def power(d, seed):
    """Share of bootstrap experiments of SESSIONS sessions with sign test p < 0.05 (DiD < 0)."""
    k_crit = next(k for k in range(SESSIONS + 1) if _sign_p(k, SESSIONS) < 0.05)
    rng = np.random.default_rng([SEED, seed, len(d)])
    neg = (np.asarray(d) < 0)
    k = neg[rng.integers(0, len(d), (BOOT, SESSIONS))].sum(1)
    return float((k >= k_crit).mean())


def power_table(sim):
    out = {}
    for cond in ("hidden", "coupled"):
        for n in COUNTS:
            d = [r["did"] for r in sim if r["cond"] == cond and r["silenced_trials"] == n]
            if not d:
                continue
            ctl = [r["light"]["control"] for r in sim if r["cond"] == cond and r["silenced_trials"] == n]
            out[f"{cond}_{n}"] = {"cond": cond, "silenced_trials": n, "power": power(d, n),
                                  "did_median": float(np.median(d)), "did_sd": float(np.std(d)),
                                  "control_excess": float(np.median(ctl)), "sims": len(d)}
    return out


def predictions(res):
    from scipy.stats import spearmanr
    a = res["replication"]
    S = np.array([r["S"] for r in a])
    P = np.array([r["S_pred"] for r in a])
    V = np.array([r["virtual"] for r in a])
    D = np.array([r["drift"] for r in a])
    n = len(S)
    rho = float(spearmanr(S, P)[0])
    pw = res["power"]
    hid = {k: pw[f"hidden_{k}"]["power"] for k in COUNTS if f"hidden_{k}" in pw}
    fp = {k: pw[f"coupled_{k}"]["power"] for k in FP_COUNTS if f"coupled_{k}" in pw}
    enough = [k for k, v in hid.items() if v >= 0.8]
    return [
        ("E1", "median S >= 0.005 and S > 0 in >= 70%", np.median(S) >= 0.005 and (S > 0).sum() >= 0.7 * n,
         f"median {np.median(S):+.4f}, {(S > 0).sum()}/{n} positive"),
        ("E2", "Spearman(S, S_pred) >= 0.7", rho >= 0.7, f"rho {rho:.2f}, median S_pred {np.median(P):+.4f}"),
        ("E3", "virtual |S| <= 0.01 in >= 80%", (np.abs(V) <= 0.01).sum() >= 0.8 * n,
         f"{(np.abs(V) <= 0.01).sum()}/{n}"),
        ("E4", "beyond drift S > 0 in >= 70%", (D > 0).sum() >= 0.7 * n,
         f"{(D > 0).sum()}/{n} positive, median {np.median(D):+.4f}"),
        ("P1", "power < 0.8 at 40 silenced trials", hid.get(40, 1.0) < 0.8, f"power {hid.get(40, float('nan')):.2f}"),
        ("P2", "power >= 0.8 reached at <= 320 silenced trials", bool(enough),
         ", ".join(f"{k}: {v:.2f}" for k, v in hid.items())),
        ("P3", "coupled false positives <= 0.1 at 160 and 320", bool(fp) and all(v <= 0.1 for v in fp.values()),
         ", ".join(f"{k}: {v:.2f}" for k, v in fp.items())),
    ]


def analyze(res) -> str:
    lines = ["=" * 78, "Experiment 73: ALM's share in an independent dataset, and the silencing test's power", "",
             "  Part A, Economo et al. 2018 (1.5 s delay before the go cue):",
             "    session                              units trials   S       S_pred   floor    virtual  beyond drift"]
    for r in res["replication"]:
        lines.append(f"    {r['session'][:36]:36} {r['units']:4d} {r['trials']:5d}  {r['S']:+.4f}  {r['S_pred']:+.4f}  "
                     f"{r['floor']:+.4f}  {r['virtual']:+.4f}  {r['drift']:+.4f}")
    lines += ["", f"  Part B, power for {SESSIONS} sessions (input load {LOAD}):",
              "    network   silenced  control excess  DiD median (sd)     power"]
    for v in res["power"].values():
        lines.append(f"    {v['cond']:9} {v['silenced_trials']:6d}    {v['control_excess']:+.4f}        "
                     f"{v['did_median']:+.4f} ({v['did_sd']:.4f})   {v['power']:.2f}")
    lines += ["", "  Pre-registered predictions:"]
    for k, desc, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {k} {desc} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    jobs = [("hidden", s, n) for n in COUNTS for s in SEEDS]
    jobs += [("coupled", s, n) for n in FP_COUNTS for s in SEEDS]
    sessions = eligible()
    if quick:
        jobs, sessions = [j for j in jobs if j[1] < 303 and j[2] <= 160], sessions[:2]
    jobs.sort(key=lambda j: -j[2])
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rep = ex.map(_replicate, sessions)
        sim = list(ex.map(_sim, jobs))
        rep = list(rep)
    res = {"replication": rep, "sim": sim, "power": power_table(sim)}
    (out_dir / "e73_alm_replication_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e73_figures import make_figures
    make_figures(res, out_dir / "e73_figures", text)
    return text


if __name__ == "__main__":
    main()
