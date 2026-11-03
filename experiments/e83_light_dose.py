"""Experiment 83: a dose-response. Does a school's interaction share rise with the light the fish see by?

e80 switched vision off in pairs and triads. Here interaction is graded: schools of about 50 rummy-nose tetras swim
for an hour at one of eight fixed light levels, from darkness to 900 lux (Lafoux et al. 2023; lafoux_data). Lafoux
found no cohesive school in darkness and cohesion rising with light.

Identities are not kept for long: FastTrack gives 140-1,500 track ids per recording for about 50 fish (median track
36 s to 13 min, 5-23% of frames missing within tracks, most in darkness). A twin built from each fish's own history
is impossible; the memory gap (e67) needs only short-lag statistics, which tracks supply:
  o(k)  member autocovariance from pairs of frames k apart within a track;
  c(k)  cross-covariance between fish from pairs of different tracks that coexist in time (so certainly different
        fish; tracks that end before another starts are excluded, since the later one may be the same fish);
  g(k)  the school's summed velocity, n o(k) + n (n - 1) c(k), with n = 50.
S = s_pred(o, g) with e80's features (planar velocity at 10 Hz, horizon 0.5 s, memgap.default_windows(5)).
Control: a virtual school with each track moved to a random start within its recording (no interaction left).

Development (simulations only, seeds 831-832): e80's simulated tetras, 50 fish, coupling 1/49 per neighbour, seeing
neighbours within a range r of the arena radius (r = 0.05, 0.1, 0.2, 0.4) or all of them, and loners. Tracks cut at
exponential intervals (mean 3,000, 1,000 or 350 frames) and resumed under a new id after 1-5 frames, with 5%, 10% or
25% of frames missing within tracks. Findings:
  * a first estimator counted the frames before and after a break as different fish: shares above 1. Restricting
    cross pairs to coexisting tracks fixed it;
  * 10-minute schools: within 0.03 of the full-identity gap, virtual schools up to 0.033 (heavy fragmentation);
  * 50-minute schools (as recorded): within 0.016 of the full-identity gap (0.6 against 0.59 when all fish are seen),
    within 0.003 below that; loners <= 0.001, virtual schools <= 0.002;
  * the share stays near 0 up to r 0.2 (<= 0.004), 0.016-0.019 at r 0.4 and 0.57-0.61 with all fish seen: a sharp
    rise once fish see across the school.

Pre-registered predictions:
  Simulations (fresh seeds 833-834, 50 minutes, each fragmentation level):
  S1  The track estimator is within 0.02 of the full-identity gap in every school.
  S2  Loners and every virtual school: |S| <= 0.005.
  S3  Order: all seen > r 0.4 > the larger of loners and r 0.05, in every seed and fragmentation level.
  Real schools (15 recordings; E1-0 replicate 0 excluded by lafoux_data's label rule):
  R1  Dose-response: Spearman correlation of light level and S > 0, one-sided p < 0.05.
  R2  Both dark recordings (E = 0) below every recording at E >= 0.3.
  R3  Virtual schools: |S| <= 0.02 in >= 80% of recordings.
  R4  Every recording at E >= 0.3 above the largest virtual-school S.
  R5  Dim light already works: both recordings at E = 0.05 (45 lux) above both dark ones.
  R6  Size: mean S at E >= 0.3 at least 3 times the dark mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import e80_intervention as e80
from experiments.e67_memory_gap import s_pred
from experiments.memgap import memories

SEED = 83
N = 50
MAXLAG = e80.SPAN
SIM_SEEDS = (833, 834)
MINUTES = 50
K = 1 / 49
SIM_CONDS = (("loners", "loners", 0.15), ("r0.05", "dark", 0.05), ("r0.1", "dark", 0.1), ("r0.2", "dark", 0.2),
             ("r0.4", "dark", 0.4), ("all", "bright", 0.15))
FRAG = ((3000, 0.05), (1000, 0.10), (350, 0.25))


# ---------------------------------------------------------------- the memory gap from tracks

def track_memories(tracks, t, n=N, maxlag=MAXLAG):
    """tracks [(start, (L, 2) with NaN rows)] over t frames. o(k) within tracks; c(k) from pairs of tracks that
    coexist in time; group g(k) = n o(k) + n (n - 1) c(k)."""
    mean = np.nanmean(np.concatenate([f for _, f in tracks]), 0)
    s, m = np.zeros((t, 2)), np.zeros(t)
    self_, comm = np.zeros(maxlag + 1), np.zeros(maxlag + 1)
    adj, adjn = np.zeros(maxlag + 1), np.zeros(maxlag + 1)
    us = []
    for a, f in tracks:
        ok = ~np.isnan(f[:, 0])
        u = np.where(ok[:, None], f - mean, 0.0)
        okf = ok.astype(float)
        us.append((a, a + len(u), u, okf))
        s[a:a + len(u)] += u
        m[a:a + len(u)] += okf
        for k in range(min(maxlag + 1, len(u))):
            self_[k] += (u[: len(u) - k] * u[k:]).sum()
            comm[k] += (okf[: len(u) - k] * okf[k:]).sum()
    starts = np.array([x[0] for x in us])
    order = np.argsort(starts, kind="stable")
    for a, e, u, ok in us:
        lo, hi = np.searchsorted(starts[order], [e, e + maxlag])
        for j in order[lo:hi]:
            b, _, w, okw = us[j]
            for k in range(b - e + 1, maxlag + 1):
                x0, x1 = max(a, b - k), min(e, b + len(w) - k)
                if x1 > x0:
                    adj[k] += (u[x0 - a:x1 - a] * w[x0 + k - b:x1 + k - b]).sum()
                    adjn[k] += (ok[x0 - a:x1 - a] * okw[x0 + k - b:x1 + k - b]).sum()
    o, c = np.zeros(maxlag + 1), np.zeros(maxlag + 1)
    for k in range(maxlag + 1):
        ss, mm = (s[: t - k] * s[k:]).sum(), (m[: t - k] * m[k:]).sum()
        o[k] = self_[k] / comm[k]
        c[k] = (ss - self_[k] - adj[k]) / (mm - comm[k] - adjn[k])
    return o, n * o + n * (n - 1) * c


def gap(o, g):
    return float(s_pred(o, g, e80.OFFS, e80.F, e80.Y))


def shift_tracks(tracks, t, rng):
    """Virtual school: each track moved to a random start within the recording."""
    return [(int(rng.integers(0, t - len(v) + 1)), v) for _, v in tracks]


def share(tracks, t, rng):
    return {"S": gap(*track_memories(tracks, t)), "S_virtual": gap(*track_memories(shift_tracks(tracks, t, rng), t))}


# ---------------------------------------------------------------- simulations

def fragment(v, rng, mean_len, drop, regap=(1, 6)):
    """v (T, n, 2) -> tracks: each fish's track breaks at exponential intervals and resumes under a new id after
    1-5 frames; a fraction drop of frames is missing within tracks."""
    t, n = v.shape[:2]
    out = []
    for i in range(n):
        a = 0
        while a < t:
            b = min(t, a + max(2, int(rng.exponential(mean_len))))
            seg = v[a:b, i].copy()
            seg[rng.random(len(seg)) < drop] = np.nan
            out.append((a, seg))
            a = b + int(rng.integers(*regap))
    return out


def run_sims(seed, quick=False):
    from experiments.intervention_data import velocity
    rng = np.random.default_rng([SEED, seed])
    rows = []
    for label, rule, r in SIM_CONDS:
        pos = e80.simulate(N, 10 if quick else MINUTES, rule, rng, k=0 if rule == "loners" else K,
                           p=dict(e80.SIM, dark_range=r))[0]
        v = velocity(pos)
        true = gap(*memories([v], MAXLAG))
        for ml, dr in FRAG:
            rows.append({"seed": seed, "cond": label, "mean_len": ml, "drop": dr, "true": true,
                         **share(fragment(v, rng, ml, dr), len(v), rng)})
    return rows


# ---------------------------------------------------------------- real schools

def run_real(quick=False):
    from experiments import lafoux_data as ld
    rng = np.random.default_rng([SEED, 1])
    rows = []
    recs = ld.recordings()
    for name, light, meta, used in recs[:: 4] if quick else recs:
        if not used:
            rows.append({"rec": name, "light": light, "used": False, "meta_light": meta["light_intensity"]})
            continue
        t, tr = ld.tracks(name)
        miss = float(np.mean(np.concatenate([np.isnan(v[:, 0]) for _, v in tr])))
        rows.append({"rec": name, "light": light, "used": True, "rep": int(meta["replicate"]), "tracks": len(tr),
                     "median_track": float(np.median([len(v) for _, v in tr])), "missing": miss, **share(tr, t, rng)})
    return rows


# ---------------------------------------------------------------- predictions

def predictions(res):
    sim = res["sim"]
    real = [r for r in res["real"] if r["used"]]
    col = lambda rs, k: np.array([r[k] for r in rs], float)  # noqa: E731
    out = []
    dev = np.abs(col(sim, "S") - col(sim, "true"))
    out.append(("S1", "simulations: track estimator within 0.02 of the full-identity gap, every school", dev.max() <= 0.02,
                f"max deviation {dev.max():.4f} ({len(sim)} schools)"))
    nulls = [r for r in sim if r["cond"] == "loners"]
    m2 = max(np.abs(col(nulls, "S")).max(), np.abs(col(sim, "S_virtual")).max())
    out.append(("S2", "simulations: loners and every virtual school |S| <= 0.005", m2 <= 0.005, f"max {m2:.4f}"))
    ok3 = []
    for seed in {r["seed"] for r in sim}:
        for ml, dr in FRAG:
            g = {r["cond"]: r["S"] for r in sim if r["seed"] == seed and r["mean_len"] == ml and r["drop"] == dr}
            ok3.append(g["all"] > g["r0.4"] > max(g["loners"], g["r0.05"]))
    out.append(("S3", "simulations: all seen > r 0.4 > max(loners, r 0.05), every seed and fragmentation", all(ok3),
                f"{sum(ok3)}/{len(ok3)}"))

    from scipy.stats import spearmanr
    rho, p2 = spearmanr(col(real, "light"), col(real, "S"))
    p = p2 / 2 if rho > 0 else 1 - p2 / 2
    out.append(("R1", "dose-response: Spearman(light, S) > 0, one-sided p < 0.05", rho > 0 and p < 0.05,
                f"rho {rho:+.2f}, p {p:.4f}, {len(real)} recordings"))
    dark = col([r for r in real if r["light"] == 0], "S")
    lit = col([r for r in real if r["light"] >= 0.3], "S")
    out.append(("R2", "both dark recordings below every recording at E >= 0.3", dark.max() < lit.min(),
                f"dark {np.round(dark, 4).tolist()}, lit min {lit.min():+.4f}"))
    v = col(real, "S_virtual")
    r3 = (np.abs(v) <= 0.02).mean()
    out.append(("R3", "virtual schools |S| <= 0.02 in >= 80% of recordings", r3 >= 0.8,
                f"{r3:.0%}; range {v.min():+.4f} to {v.max():+.4f}"))
    out.append(("R4", "every recording at E >= 0.3 above the largest virtual S", lit.min() > v.max(),
                f"lit min {lit.min():+.4f}, virtual max {v.max():+.4f}"))
    dim = col([r for r in real if r["light"] == 0.05], "S")
    out.append(("R5", "dim light works: both E = 0.05 recordings above both dark ones", dim.min() > dark.max(),
                f"E 0.05 {np.round(dim, 4).tolist()}, dark {np.round(dark, 4).tolist()}"))
    out.append(("R6", "size: mean S at E >= 0.3 at least 3x the dark mean", lit.mean() >= 3 * dark.mean(),
                f"lit mean {lit.mean():+.4f}, dark mean {dark.mean():+.4f}"))
    return out


def analyze(res) -> str:
    lines = ["Experiment 83: does the share rise with light? Schools of ~50 tetras at fixed light levels", "",
             "  Simulated schools (50 fish, 50 minutes): full-identity gap, track estimate, virtual school",
             "  seed cond    fragmentation      true      tracks    virtual"]
    for r in res["sim"]:
        lines.append(f"  {r['seed']}  {r['cond']:7} L{r['mean_len']:<5d} drop {r['drop']:<5} {r['true']:+.4f}   "
                     f"{r['S']:+.4f}   {r['S_virtual']:+.4f}")
    lines += ["", "  Real schools (Lafoux et al. 2023):",
              "  light  rep  tracks  median track (s)  missing   S         virtual"]
    for r in res["real"]:
        if not r["used"]:
            lines.append(f"  {r['light']:<5}  excluded: {r['rec']} (metadata says {r['meta_light']})")
            continue
        lines.append(f"  {r['light']:<5}  {r['rep']}    {r['tracks']:<6d}  {r['median_track'] / 10:8.0f}          "
                     f"{r['missing']:.3f}    {r['S']:+.4f}   {r['S_virtual']:+.4f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    res = {"sim": [r for s in SIM_SEEDS[: 1 if quick else None] for r in run_sims(s, quick)], "real": run_real(quick)}
    (out_dir / "e83_light_dose_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e83_figures import make_figures
    make_figures(res, out_dir / "e83_figures", text)
    return text


if __name__ == "__main__":
    main()
