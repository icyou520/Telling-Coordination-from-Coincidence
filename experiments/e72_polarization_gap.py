"""Experiment 72: a quadratic memory gap for polarization (can the equation get the size right on fish?).

e68: the linear memory gap separated every real fish school from its virtual groups but ranked real schools the
wrong way (Spearman -0.65). Adult fish have the highest measured S_proj (0.64-0.89) and the lowest predictions.
Diagnosis, made before any of the numbers below were computed: the linear equation centres each fish's heading,
so it ignores the *level* of polarization. A no-interaction twin predicts that a tightly aligned school relaxes
towards disorder (phi^2 -> 1/N), and that error is most of the measured share for a polarized 8-fish school.

The quadratic memory gap keeps every assumption of the linear one except linearity in the group variable:
  * each fish's twin forecast uses only its own memory: m_i = mu_i + sum_w beta_w(o) * (window sums of its centred
    heading), with beta(o) the e68 memory weights (same windows, 1 s ahead);
  * the fish are independent under the twin, so its forecast of polarization squared is exact given m:
      phi0^2 = (N + |sum_i m_i|^2 - sum_i |m_i|^2) / N^2;
  * group law G: least squares of phi^2(t) on the school's own past, [1, phi_w, phi_w^2, phi_w^3] for the three
    windows; G0_proj: the same regression fitted to phi0^2; both scored on the real phi^2 (S_proj, e30);
  * cross-fitted over 3 folds of interleaved 60 s blocks.
No fish model (walls, speed, turning) is fitted: only the member memory curve.

Development on simulation only (seeds 0-1; persistent independent walkers and heading-aligning agents, N = 8 and
60): independent groups gave |S_q| <= 0.012; aligning groups 0.05-0.52 (N 8) and 0.31-0.87 (N 60), rising with
alignment strength 0.02-0.1 (registered checks use 0.1).

Compared with e33's S_proj (8 adult-fish recordings and their virtual groups) and e38's (9 zebrafish recordings,
9 virtual groups), as in e68. Simulated checks use fresh seeds 10-14.

Predictions, fixed before running:
  Q1  Simulated independent walkers (N 8 and 60, 5 seeds each): |S_q| <= 0.03 in >= 9/10.
  Q2  Simulated aligning groups (N 8 and 60, 5 seeds each): S_q >= 0.2 in >= 9/10.
  Q3  Virtual fish groups: |S_q| <= 0.05 in >= 80% of the 17.
  Q4  Real schools: S_q >= virtual + 0.1 in every recording (zebrafish: mean of its size's virtual groups).
  Q5  Within real schools, the size is ranked: Spearman(S_q, measured S_proj) >= 0.5 (e68's F4 failure).
  Q6  The size is right: median |S_q - measured S_proj| over the 17 real schools <= 0.15.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.e67_memory_gap import _toeplitz, design
from experiments.e68_fish_memory_gap import WINDOWS, memories

BLOCK = 600
FOLDS = 3
SIM_SEEDS = tuple(range(10, 15))


def unit(theta):
    return np.stack([np.cos(theta), np.sin(theta)], -1)


def _window_sums(c, offs, f):
    """c (T, ...) centred series; returns (rows, w, ...) window sums at targets t = rows, and the rows."""
    lo, hi = int(offs.min()), int(offs.max())
    rows = np.arange(-lo, len(c) - hi)
    sums = np.stack([sum(c[rows + o] for o in offs[f[w] > 0]) for w in range(len(f))], 1)
    return sums, rows


def _ols(xtr, ytr, xte):
    coef, *_ = np.linalg.lstsq(xtr, ytr, rcond=None)
    return xte @ coef


def evaluate_q(theta):
    theta = theta[np.isfinite(theta).all(1)]
    t, n = theta.shape
    u = unit(theta)
    mu = u.mean(0)
    c = u - mu
    offs, f, y = design(WINDOWS, [0])
    o, _ = memories(theta)
    mo = _toeplitz(o, offs - offs.min())
    beta = np.linalg.solve(f @ mo @ f.T, f @ mo @ y)
    sums, rows = _window_sums(c, offs, f)
    m = mu[None] + np.einsum("w,rwnd->rnd", beta, sums)
    phi0 = (n + (m.sum(1) ** 2).sum(-1) - (m ** 2).sum((1, 2))) / n ** 2
    phi = np.sqrt((u.sum(1) ** 2).sum(-1)) / n
    target = phi[rows] ** 2
    pw = np.stack([sum(phi[rows + o] for o in offs[f[w] > 0]) / f[w].sum() for w in range(len(f))], 1)
    feats = np.column_stack([np.ones(len(rows))] + [pw ** p for p in (1, 2, 3)])
    fold = (np.arange(len(rows)) // BLOCK) % FOLDS
    eg, e0 = np.empty(len(rows)), np.empty(len(rows))
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        eg[te] = target[te] - _ols(feats[tr], target[tr], feats[te])
        e0[te] = target[te] - _ols(feats[tr], phi0[tr], feats[te])
    return {"S_q": float(1 - (eg ** 2).sum() / (e0 ** 2).sum()), "phi_mean": float(phi.mean()), "n_steps": int(t)}


def simulate(kind, n, seed, steps=20000, align=0.1):
    """Headings at 10 Hz: persistent turning (AR(1) turn rate) plus, for 'aligning', a pull towards the mean."""
    rng = np.random.default_rng([72, n, seed, kind == "aligning"])
    th = rng.uniform(-np.pi, np.pi, n)
    w = np.zeros(n)
    out = np.empty((steps, n))
    for s in range(steps):
        out[s] = th
        w = 0.9 * w + 0.05 * rng.normal(size=n)
        pull = 0.0
        if kind == "aligning":
            mean = np.arctan2(np.sin(th).sum(), np.cos(th).sum())
            pull = align * np.sin(mean - th)
        th = th + w + pull
    return out


def _sim(job):
    kind, n, seed = job
    return {"kind": kind, "n": n, "seed": seed, **evaluate_q(simulate(kind, n, seed))}


def _adult(args):
    from experiments.e31_real_fish import kinematics, virtual
    from experiments.fish_data import load
    n, rec = args
    pos = load(n, rec)
    return {"set": "adult", "n": n, "rec": rec, "real": evaluate_q(kinematics(pos)[1]),
            "virtual": evaluate_q(kinematics(virtual(pos))[1])}


def _zebra_real(args):
    from experiments import e38_zebrafish as z
    n, rec = args
    pos, _, dt = z.load(n, rec)
    return {"set": "zebrafish", "n": n, "rec": rec, "real": evaluate_q(z.kinematics(pos, dt)[1])}


def _zebra_virtual(args):
    from experiments import e38_zebrafish as z
    n, m = args
    loaded = [z.load(n, r) for r in z.RECS]
    dt = float(np.mean([l[2] for l in loaded]))
    length = min(len(l[0]) for l in loaded)
    per = n // 3 + 1
    pos = np.empty((length, n, 2))
    for i in range(n):
        pos[:, i] = np.roll(loaded[(i + m) % 3][0][:length, i], (i // 3) * length // per, axis=0)
    return {"set": "zebrafish", "n": n, "m": m, "virtual": evaluate_q(z.kinematics(pos, dt)[1])}


def table(res):
    from experiments.e68_fish_memory_gap import measured
    adult, zr, zv = measured()
    e68 = json.loads(Path("results/e68_fish_memory_gap_raw.json").read_text())
    lin = {("adult", r["n"], r["rec"], k): r[k]["S_pred"] for r in e68["adult"] for k in ("real", "virtual")}
    lin.update({("zebrafish", r["n"], r["rec"], "real"): r["real"]["S_pred"] for r in e68["zebra_real"]})
    lin.update({("zebrafish", r["n"], r["m"], "virtual"): r["virtual"]["S_pred"] for r in e68["zebra_virtual"]})
    rows = []
    for r in res["adult"]:
        mr, mv = adult[(r["n"], r["rec"])]
        rows.append(("adult", f"N{r['n']} rec{r['rec']}", "real", r["real"], mr, lin[("adult", r["n"], r["rec"], "real")]))
        rows.append(("adult", f"N{r['n']} rec{r['rec']}", "virtual", r["virtual"], mv,
                     lin[("adult", r["n"], r["rec"], "virtual")]))
    for r in res["zebra_real"]:
        rows.append(("zebrafish", f"N{r['n']} rec{r['rec']}", "real", r["real"], zr[(r["n"], r["rec"])],
                     lin[("zebrafish", r["n"], r["rec"], "real")]))
    for r in res["zebra_virtual"]:
        rows.append(("zebrafish", f"N{r['n']} m{r['m']}", "virtual", r["virtual"], zv[(r["n"], r["m"])],
                     lin[("zebrafish", r["n"], r["m"], "virtual")]))
    return rows


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def predictions(res):
    sim = res["sim"]
    ind = [r["S_q"] for r in sim if r["kind"] == "independent"]
    ali = [r["S_q"] for r in sim if r["kind"] == "aligning"]
    rows = table(res)
    virt = [r for r in rows if r[2] == "virtual"]
    real = [r for r in rows if r[2] == "real"]
    q3 = sum(abs(r[3]["S_q"]) <= 0.05 for r in virt)
    q4 = sum(r["real"]["S_q"] >= r["virtual"]["S_q"] + 0.1 for r in res["adult"])
    for r in res["zebra_real"]:
        vm = np.mean([v["virtual"]["S_q"] for v in res["zebra_virtual"] if v["n"] == r["n"]])
        q4 += r["real"]["S_q"] >= vm + 0.1
    q5 = _spearman([r[3]["S_q"] for r in real], [r[4] for r in real])
    q6 = float(np.median([abs(r[3]["S_q"] - r[4]) for r in real]))
    return [
        ("Q1", "simulated independent walkers: |S_q| <= 0.03 in >= 9/10", sum(abs(s) <= 0.03 for s in ind) >= 9,
         f"{sum(abs(s) <= 0.03 for s in ind)}/{len(ind)}, max |S_q| {max(abs(s) for s in ind):.3f}"),
        ("Q2", "simulated aligning groups: S_q >= 0.2 in >= 9/10", sum(s >= 0.2 for s in ali) >= 9,
         f"{sum(s >= 0.2 for s in ali)}/{len(ali)}, range {min(ali):.2f}-{max(ali):.2f}"),
        ("Q3", "virtual fish groups: |S_q| <= 0.05 in >= 80%", q3 >= 0.8 * len(virt), f"{q3}/{len(virt)}"),
        ("Q4", "real schools: S_q >= virtual + 0.1 in every recording", q4 == len(real), f"{q4}/{len(real)}"),
        ("Q5", "within real schools: Spearman(S_q, S_proj) >= 0.5", q5 >= 0.5, f"rho {q5:+.2f} (linear e68: -0.65)"),
        ("Q6", "size: median |S_q - S_proj| over real schools <= 0.15", q6 <= 0.15, f"{q6:.3f}"),
    ]


def analyze(res) -> str:
    lines = ["=" * 78, "Experiment 72: a quadratic memory gap for polarization", "", "  Simulations:"]
    for kind in ("independent", "aligning"):
        for n in (8, 60):
            v = [r["S_q"] for r in res["sim"] if r["kind"] == kind and r["n"] == n]
            lines.append(f"    {kind:12} N {n:3d}  S_q " + " ".join(f"{s:+.3f}" for s in v))
    lines += ["", "  set        group          kind      S_q     measured S_proj   linear S_pred (e68)   mean phi"]
    for s, g, k, e, m, lin in table(res):
        lines.append(f"  {s:10} {g:14} {k:8} {e['S_q']:+.3f}   {m:+.3f}             {lin:+.3f}               "
                     f"{e['phi_mean']:.2f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    from concurrent.futures import ProcessPoolExecutor
    from experiments.e38_zebrafish import RECS, SIZES
    from experiments.fish_data import RECORDINGS
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    sims = [(k, n, s) for k in ("independent", "aligning") for n in (8, 60) for s in SIM_SEEDS]
    adult = [(n, r) for n, recs in RECORDINGS.items() for r in recs]
    zr = [(n, r) for n in SIZES for r in RECS]
    zv = [(n, m) for n in SIZES for m in range(3)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        res = {"sim": list(pool.map(_sim, sims)), "adult": list(pool.map(_adult, adult)),
               "zebra_real": list(pool.map(_zebra_real, zr)), "zebra_virtual": list(pool.map(_zebra_virtual, zv))}
    (out_dir / "e72_polarization_gap_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e72_figures import make_figures
    make_figures(res, out_dir / "e72_figures", text)
    return text


if __name__ == "__main__":
    main()
