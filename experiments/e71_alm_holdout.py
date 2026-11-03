"""Experiment 71: ALM's small coordination share, confirmed on unseen sessions and tested against silencing.

e70 (exploratory) found that with all ~270 control trials, ALM in the 1.5 s before the go cue has a small share
beyond each neuron's side-specific response: measured 0.014 (median of e64's 21 sessions), memory-gap prediction
0.017, Spearman 0.94. That was found after looking, on sessions analysed before. Two questions follow.

Part A, confirmation on unseen data. The DANDI 000221 ALM sessions with >= 20 units that e64/e65 did not use
(fewer than 20 silenced trials), 19 sessions, 136-371 control trials each. No share of any kind has been computed on
them. Pipeline exactly as e70: control trials, bins 50-199 (1.5 s before the cue, 10 ms), counts minus
cross-fitted side-specific means (e67 residualize, 5 folds), e62 history windows, next 50 ms; measured S = e67's
cross-fitted linear S_proj (rows 99-193, step 2), S_pred = memory-gap tool.
  Virtual: each neuron's trials permuted independently within side, then the same pipeline (measured S).
  Beyond drift: after residualizing, each neuron's count minus its mean in this window over the 10 trials before
  and after (excluding the trial itself), removing slow session drift that every neuron shares.

Part B, does the share need the midbrain input? With ~40 silenced trials the fitted twins of e65 were too noisy
(about +-0.03). The memory gap needs no fitting. On e64's 21 sessions: S_pred minus its trial-permutation floor
("excess"), for silenced trials and for control trials subsampled to the silenced count (5 draws, averaged), in
the light window before the cue (bins 140-199) and in a baseline window before the light (bins 60-119), e64's
short history windows. Effect = difference-in-differences (silenced - control in light) - (same in baseline).

Developed on e64's simulated networks only (development seeds 10-30): independent and coupled networks gave
|DiD| <= 0.025, a hidden input switched off -0.18 to -0.43; a weak input (load 0.1, control share about 0.025)
was detected over 21 sessions (16/21 negative, p 0.013), a weaker one (share 0.001) was not.
Simulations here use fresh seeds 30-39 (and 30-50 for the power check).
Part A's pipeline on development simulations (30 neurons): independent networks measured -0.004 at 150 trials and
-0.0015 at 300 (cross-fitting bias, conservative for A1); a weak input 0.002 and 0.011.

Predictions, fixed before running:
  C1  Independent networks: |DiD| <= 0.03 in >= 8/10 seeds.
  C2  Hidden input switched off: DiD <= -0.1 in >= 8/10.
  C3  Coupled networks (cue input silenced, coupling intact): |DiD| <= 0.03 in >= 8/10.
  C4  Power: weak input (load 0.1), 21 simulated sessions: DiD < 0 with one-sided sign test p < 0.05.
  A1  Holdout: median measured S >= 0.005 and S > 0 in >= 14/19 sessions.
  A2  Holdout: Spearman(measured S, S_pred) >= 0.7.
  A3  Holdout virtual groups: |S| <= 0.01 in >= 16/19.
  A4  Holdout beyond drift: S > 0 in >= 14/19 sessions.
  B1  Before the light nothing differs: |median excess(baseline, silenced - control)| <= 0.01.
  B2  The share needs the midbrain input: median DiD < 0, one-sided sign test p < 0.05.
Reading of B2, fixed in advance: if B2 fails and the median control excess in the light window is < 0.02, the test
is underpowered (C4) and inconclusive; if it fails with control excess >= 0.02 and |median DiD| <= 0.005, the
share does not depend on the midbrain input at this resolution.
"""

from __future__ import annotations

import inspect
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments import e64_input_silencing as e64
from experiments.e62_neural_share_repaired import H as H62, HIST as HIST62
from experiments.memgap import screen

SEED = 71
MIN_UNITS = 20
A_BINS = (50, 200)
A_ROWS = np.arange(49, A_BINS[1] - A_BINS[0] - H62, 2)
LIGHT, BASE = (140, 200), (60, 120)
DRAWS = 5
DRIFT = 10
SIM_SEEDS = tuple(range(30, 40))
POWER_SEEDS = tuple(range(30, 51))
WEAK = 0.1

_src = inspect.getsource(e64.simulate)
_src = _src.replace("def simulate(cond, seed, n=40, trials=80):",
                    "def simulate(cond, seed, n=40, trials=80, load_mean=0.4):")
_src = _src.replace('rng.normal(0.4, 0.15, n) * (cond == "hidden")',
                    'rng.normal(load_mean, load_mean * 0.375, n) * (cond == "hidden")')
_ns = dict(vars(e64))
exec(_src, _ns)
simulate = _ns["simulate"]


def _fold(nt, seed):
    return np.random.default_rng([SEED, seed, nt]).permutation(np.arange(nt) % 5)


def excess(x, left, win, seed):
    """S_pred minus its trial-permutation floor, e64 history windows, next 50 ms."""
    from experiments.e67_memory_gap import residualize
    nt = x.shape[1]
    xr = residualize(x.astype(float), left.astype(int), _fold(nt, seed))[:, :, win[0]:win[1]]
    r = screen([xr[:, t].T for t in range(nt)], e64.HIST, target=range(1, e64.H + 1), baseline="permute",
               seed=seed)
    return r["S_pred"] - r["virtual"], r["S_pred"]


def did(x, silenced, left, seed):
    rng = np.random.default_rng([SEED, 9, seed])
    sil, ctl = np.flatnonzero(silenced), np.flatnonzero(~silenced)
    out = {}
    for name, win in (("light", LIGHT), ("baseline", BASE)):
        s = excess(x[:, sil], left[sil], win, seed)[0]
        c = np.mean([excess(x[:, d], left[d], win, seed + k)[0]
                     for k, d in enumerate(np.sort(rng.choice(ctl, len(sil), replace=False)) for _ in range(DRAWS))])
        out[name] = {"silenced": float(s), "control": float(c)}
    out["did"] = (out["light"]["silenced"] - out["light"]["control"]) - (
        out["baseline"]["silenced"] - out["baseline"]["control"])
    return out


def _sim(job):
    cond, seed, load = job
    x, silenced, left = simulate(cond, seed, load_mean=load)
    return {"cond": cond if load == 0.4 or cond != "hidden" else "weak", "seed": seed, **did(x, silenced, left, seed)}


def drift_removed(x):
    """x (units, trials, bins): minus each neuron's window mean over the DRIFT trials either side (not itself)."""
    m = x.mean(2)
    nt = m.shape[1]
    d = np.empty_like(m)
    for t in range(nt):
        idx = [k for k in range(max(0, t - DRIFT), min(nt, t + DRIFT + 1)) if k != t]
        d[:, t] = m[:, idx].mean(1)
    return x - d[:, :, None]


def _holdout(name):
    from experiments import inagaki_data as idata
    from experiments.e67_memory_gap import evaluate, residualize, trial_design
    s = idata.load(name)
    c = ~s["silenced"]
    x, left = s["x"][:, c].astype(float), s["left"][c]
    fold = _fold(x.shape[1], 0)
    offs, f, y, _ = trial_design()

    def measure(xx):
        xr = residualize(xx, left.astype(int), fold)[:, :, A_BINS[0]:A_BINS[1]]
        return xr, evaluate(xr, offs, f, y, A_ROWS, fold)["S"]

    xr, real = measure(x)
    pred = screen([xr[:, t].T for t in range(xr.shape[1])], HIST62, target=range(1, H62 + 1),
                  baseline="permute")
    rng = np.random.default_rng([SEED, 3, x.shape[0]])
    perm = np.empty_like(x)
    for i in range(x.shape[0]):
        p = np.arange(x.shape[1])
        for side in (True, False):
            idx = np.flatnonzero(left == side)
            p[idx] = rng.permutation(idx)
        perm[i] = x[i, p]
    _, virt = measure(perm)
    xd = residualize(x, left.astype(int), fold)[:, :, A_BINS[0]:A_BINS[1]]
    xd = drift_removed(xd)
    drift = evaluate(xd, offs, f, y, A_ROWS, fold)["S"]
    return {"session": name, "units": int(x.shape[0]), "trials": int(x.shape[1]), "S": real, "S_pred": pred["S_pred"],
            "floor": pred["virtual"], "virtual": virt, "drift": drift}


def _silencing(name):
    from experiments import inagaki_data as idata
    s = idata.load(name)
    return {"session": name, "units": int(s["x"].shape[0]), "silenced_trials": int(s["silenced"].sum()),
            **did(s["x"], s["silenced"], s["left"], len(name) + s["x"].shape[0])}


def holdout_sessions():
    from experiments import inagaki_data as idata
    used = set(e64.eligible())
    out = []
    for r in idata.sessions():
        n = r["Session_name"]
        if n not in used and idata.load(n)["x"].shape[0] >= MIN_UNITS:
            out.append(n)
    return out


def _sign_p(k, n):
    from scipy.stats import binomtest
    return float(binomtest(int(k), int(n), 0.5, alternative="greater").pvalue)


def predictions(res):
    from scipy.stats import spearmanr
    sim = res["sim"]

    def by(c):
        return np.array([r["did"] for r in sim if r["cond"] == c])

    ind, hid, cou, weak = by("independent"), by("hidden"), by("coupled"), by("weak")
    a = res["holdout"]
    S = np.array([r["S"] for r in a])
    P = np.array([r["S_pred"] for r in a])
    V = np.array([r["virtual"] for r in a])
    D = np.array([r["drift"] for r in a])
    b = res["silencing"]
    dd = np.array([r["did"] for r in b])
    base = np.median([r["baseline"]["silenced"] - r["baseline"]["control"] for r in b])
    rho = float(spearmanr(S, P)[0])
    return [
        ("C1", "independent networks: |DiD| <= 0.03 in >= 8/10", (np.abs(ind) <= 0.03).sum() >= 8,
         f"{(np.abs(ind) <= 0.03).sum()}/10"),
        ("C2", "hidden input switched off: DiD <= -0.1 in >= 8/10", (hid <= -0.1).sum() >= 8, f"{(hid <= -0.1).sum()}/10"),
        ("C3", "coupled networks: |DiD| <= 0.03 in >= 8/10", (np.abs(cou) <= 0.03).sum() >= 8,
         f"{(np.abs(cou) <= 0.03).sum()}/10"),
        ("C4", "power: weak input, 21 sessions, sign test p < 0.05", _sign_p((weak < 0).sum(), len(weak)) < 0.05,
         f"{(weak < 0).sum()}/{len(weak)} negative, p {_sign_p((weak < 0).sum(), len(weak)):.4f}"),
        ("A1", "holdout: median S >= 0.005 and S > 0 in >= 14/19", np.median(S) >= 0.005 and (S > 0).sum() >= 14,
         f"median {np.median(S):+.4f}, {(S > 0).sum()}/{len(S)} positive"),
        ("A2", "holdout: Spearman(S, S_pred) >= 0.7", rho >= 0.7, f"rho {rho:.2f}, median S_pred {np.median(P):+.4f}"),
        ("A3", "holdout virtual: |S| <= 0.01 in >= 16/19", (np.abs(V) <= 0.01).sum() >= 16,
         f"{(np.abs(V) <= 0.01).sum()}/{len(V)}"),
        ("A4", "holdout beyond drift: S > 0 in >= 14/19", (D > 0).sum() >= 14,
         f"{(D > 0).sum()}/{len(D)} positive, median {np.median(D):+.4f}"),
        ("B1", "baseline: |median silenced - control| <= 0.01", abs(base) <= 0.01, f"{base:+.4f}"),
        ("B2", "share needs the midbrain input: median DiD < 0, sign test p < 0.05",
         np.median(dd) < 0 and _sign_p((dd < 0).sum(), len(dd)) < 0.05,
         f"median {np.median(dd):+.4f}, {(dd < 0).sum()}/{len(dd)} negative, p {_sign_p((dd < 0).sum(), len(dd)):.3f}"),
    ]


def analyze(res) -> str:
    lines = ["=" * 78, "Experiment 71: ALM's small share, on unseen sessions and against silencing", ""]
    lines.append("  Simulations (difference-in-differences of the memory-gap excess):")
    for c in ("independent", "coupled", "hidden", "weak"):
        d = [r["did"] for r in res["sim"] if r["cond"] == c]
        ctl = [r["light"]["control"] for r in res["sim"] if r["cond"] == c]
        lines.append(f"    {c:12} control excess {np.median(ctl):+.3f}  DiD median {np.median(d):+.3f} "
                     f"[{min(d):+.3f}, {max(d):+.3f}]")
    lines += ["", "  Part A, holdout sessions (control trials, 1.5 s before the cue):",
              "    session        units trials   S       S_pred   floor    virtual  beyond drift"]
    for r in res["holdout"]:
        lines.append(f"    {r['session']:14} {r['units']:4d} {r['trials']:5d}  {r['S']:+.4f}  {r['S_pred']:+.4f}  "
                     f"{r['floor']:+.4f}  {r['virtual']:+.4f}  {r['drift']:+.4f}")
    lines += ["", "  Part B, silencing (memory-gap excess; control subsampled to the silenced count):",
              "    session        units sil   light sil  light ctl  base sil  base ctl   DiD"]
    for r in res["silencing"]:
        lines.append(f"    {r['session']:14} {r['units']:4d} {r['silenced_trials']:4d}  {r['light']['silenced']:+.4f}    "
                     f"{r['light']['control']:+.4f}    {r['baseline']['silenced']:+.4f}   {r['baseline']['control']:+.4f}   "
                     f"{r['did']:+.4f}")
    ctl = np.median([r["light"]["control"] for r in res["silencing"]])
    lines.append(f"    median control excess in the light window {ctl:+.4f}")
    lines += ["", "  Pre-registered predictions:"]
    preds = predictions(res)
    for k, desc, ok, detail in preds:
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {k} {desc} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    sim_jobs = [(c, s, 0.4) for c in ("independent", "coupled", "hidden") for s in SIM_SEEDS]
    sim_jobs += [("hidden", s, WEAK) for s in POWER_SEEDS]
    hold = holdout_sessions()
    sil = e64.eligible()
    if quick:
        sim_jobs, hold, sil = sim_jobs[::10], hold[:2], sil[:2]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = {"sim": list(ex.map(_sim, sim_jobs)), "holdout": list(ex.map(_holdout, hold)),
               "silencing": list(ex.map(_silencing, sil))}
    (out_dir / "e71_alm_holdout_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e71_figures import make_figures
    make_figures(res, out_dir / "e71_figures", text)
    return text


if __name__ == "__main__":
    main()
