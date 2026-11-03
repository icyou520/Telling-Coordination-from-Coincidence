"""Experiment 68: does the memory-gap equation (e67) predict the interaction share of fish schools?

e67 showed that, for linear forecasts, the interaction share is the cost of forecasting a group with its members'
memory instead of its own, and that the equation matched simulations, markets and 221 brain areas. Fish are the
hardest test: the measured shares (e33 adult fish, e38 zebrafish) used nonlinear twins (wall avoidance, persistent
turning, speed) and a cubic law in polarization, a nonlinear group variable.

Here the equation gets no twin and no polarization. Members: each fish's heading as a unit vector, every 0.1 s
(e31 / e38 kinematics). Group: the school's summed heading vector. Memories: o(k) = mean over fish of the
autocovariance of its centred heading vector (both components), g(k) = the same for the summed vector.
Forecast: 1 s ahead (10 steps, as e33 and e38), from three history windows: the heading 1 s before the target, the
mean over 1.1-1.4 s before, and the mean over 1.5-2.0 s before. S_pred is e67's formula on these windows.

Compared with: e33's S_proj for 8 adult-fish recordings (6 of 8 fish, 2 of 39) and their virtual groups; e38's
S_proj for 9 zebrafish recordings (60, 80, 100 fish) and 9 virtual groups. Those values are known; the equation's
values are not (no S_pred was computed on fish before registration).

Predictions, fixed before running:
  F1  Virtual groups: |S_pred| <= 0.03 in >= 80% of the 17.
  F2  Real schools: S_pred exceeds its virtual counterpart(s) by >= 0.05 in every real recording
      (adult fish: its own virtual group; zebrafish: the mean of the 3 virtual groups of its size).
  F3  The equation ranks the measured shares: Spearman(S_pred, measured S_proj) >= 0.7 over all 34 groups.
  F4  Within real schools only (17): Spearman(S_pred, measured S_proj) >= 0.5.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.e67_memory_gap import design, s_one, s_pred

WINDOWS = ((10, 11), (11, 15), (15, 21))


def memories(theta):
    """theta (T, N) headings. Returns member o(k) and group g(k) for k = 0..maxlag, centred unit vectors."""
    u = np.stack([np.cos(theta), np.sin(theta)], -1)
    u = u - u.mean(0, keepdims=True)
    v = u.sum(1)
    offs, _, _ = design(WINDOWS, [0])
    maxlag = int(offs.max() - offs.min())
    t = len(u)
    o = np.array([(u[: t - k] * u[k:]).sum(-1).mean() for k in range(maxlag + 1)])
    g = np.array([(v[: t - k] * v[k:]).sum(-1).mean() for k in range(maxlag + 1)])
    return o, g


def evaluate(theta):
    theta = theta[np.isfinite(theta).all(1)]
    o, g = memories(theta)
    offs, f, y = design(WINDOWS, [0])
    return {"S_pred": s_pred(o, g, offs, f, y), "S_1": s_one(o, g), "rho_m": float(o[1] / o[0]),
            "rho_g": float(g[1] / g[0]), "n_steps": int(len(theta))}


def _adult(args):
    from experiments.e31_real_fish import kinematics, virtual
    from experiments.fish_data import load
    n, rec = args
    pos = load(n, rec)
    out = {"set": "adult", "n": n, "rec": rec}
    for kind, data in (("real", pos), ("virtual", virtual(pos))):
        out[kind] = evaluate(kinematics(data)[1])
    return out


def _zebra_real(args):
    from experiments import e38_zebrafish as z
    n, rec = args
    pos, _, dt = z.load(n, rec)
    return {"set": "zebrafish", "n": n, "rec": rec, "real": evaluate(z.kinematics(pos, dt)[1])}


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
    return {"set": "zebrafish", "n": n, "m": m, "virtual": evaluate(z.kinematics(pos, dt)[1])}


def measured():
    a = json.loads(Path("results/e33_fish_projected_twin_raw.json").read_text())["real"]
    b = json.loads(Path("results/e38_zebrafish_raw.json").read_text())
    adult = {(r["n"], r["rec"]): (r["real"]["S_proj"], r["virtual"]["S_proj"]) for r in a}
    zr = {(r["n"], r["rec"]): r["real"]["S_proj"] for r in b["real"]}
    zv = {(r["n"], r["m"]): r["virtual"]["S_proj"] for r in b["virtual"]}
    return adult, zr, zv


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0]) if len(a) > 2 else np.nan


def table(res):
    adult, zr, zv = measured()
    rows = []
    for r in res["adult"]:
        m_real, m_virt = adult[(r["n"], r["rec"])]
        rows.append(("adult", f"N{r['n']} rec{r['rec']}", "real", r["real"]["S_pred"], m_real, r))
        rows.append(("adult", f"N{r['n']} rec{r['rec']}", "virtual", r["virtual"]["S_pred"], m_virt, r))
    for r in res["zebra_real"]:
        rows.append(("zebrafish", f"N{r['n']} rec{r['rec']}", "real", r["real"]["S_pred"], zr[(r["n"], r["rec"])], r))
    for r in res["zebra_virtual"]:
        rows.append(("zebrafish", f"N{r['n']} m{r['m']}", "virtual", r["virtual"]["S_pred"], zv[(r["n"], r["m"])], r))
    return rows


def predictions(res):
    rows = table(res)
    virt = [r for r in rows if r[2] == "virtual"]
    real = [r for r in rows if r[2] == "real"]
    f1 = sum(abs(r[3]) <= 0.03 for r in virt)
    f2 = 0
    for r in res["adult"]:
        f2 += r["real"]["S_pred"] >= r["virtual"]["S_pred"] + 0.05
    for r in res["zebra_real"]:
        vm = np.mean([v["virtual"]["S_pred"] for v in res["zebra_virtual"] if v["n"] == r["n"]])
        f2 += r["real"]["S_pred"] >= vm + 0.05
    f3 = _spearman([r[3] for r in rows], [r[4] for r in rows])
    f4 = _spearman([r[3] for r in real], [r[4] for r in real])
    return [
        ("F1", "virtual groups: |S_pred| <= 0.03 in >= 80%", f1 >= 0.8 * len(virt), f"{f1}/{len(virt)}"),
        ("F2", "real schools: S_pred >= virtual + 0.05 in every recording", f2 == len(real), f"{f2}/{len(real)}"),
        ("F3", "ranks the measured shares: Spearman(S_pred, S_proj) >= 0.7 over all groups", f3 >= 0.7,
         f"rho {f3:+.2f} ({len(rows)} groups)"),
        ("F4", "within real schools: Spearman >= 0.5", f4 >= 0.5, f"rho {f4:+.2f} ({len(real)} schools)"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 68: does the memory-gap equation predict the interaction share of fish schools?", "",
             "  set        group          kind      S_pred   measured S_proj   rho_m   rho_g"]
    for s, g, k, p, m, r in table(res):
        e = r[k]
        lines.append(f"  {s:10} {g:14} {k:8} {p:+.3f}    {m:+.3f}            {e['rho_m']:.3f}   {e['rho_g']:.3f}")
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
    adult = [(n, r) for n, recs in RECORDINGS.items() for r in recs]
    zr = [(n, r) for n in SIZES for r in RECS]
    zv = [(n, m) for n in SIZES for m in range(3)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        res = {"adult": list(pool.map(_adult, adult)), "zebra_real": list(pool.map(_zebra_real, zr)),
               "zebra_virtual": list(pool.map(_zebra_virtual, zv))}
    (out_dir / "e68_fish_memory_gap_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e68_figures import make_figures
    make_figures(res, out_dir / "e68_figures", text)
    return text


if __name__ == "__main__":
    main()
