"""Experiment 80: switching interaction off. Does the share fall when the experimenter removes interaction?

Every earlier result was observational. Here interaction is removed or weakened by the experimenter, on two public
datasets (intervention_data):
  light  Liu & Burbano (2025): zebrafish pairs and triads in bright and infrared-dark arenas (vision removed; the
         lateral line remains), and single fish in the same arenas, which give groups that certainly never
         interacted.
  VR     Escobedo et al. (2025): a real rummy-nose tetra with a virtual conspecific that responds to it (C1, C2) or
         ignores it (C3: C2's clone, one-way); two real fish (2R).

Member variable: planar velocity at 10 Hz. Group variable: the members' summed velocity. Horizon 0.5 s; features are
e67's windows before the target (memgap.default_windows(5)). Shares, cross-fitted over 30 s blocks in 3 folds with
purged training rows (nothing fitted sees the test rows or their lags):
  member  the twin forecasts each member from its own past with weights fitted on the members (the memory twin);
  iso     the twin forecasts each member with weights fitted on single fish of the same light condition, recorded
          alone: the measured behaviour of a fish without a partner (light data only);
  gap     the memory gap S_pred from member and group autocovariance (memgap), and S_pred,iso from single fish.
For linear laws on sums the twin forecast is already a function of the group's features, so S = S_proj.
Controls: synthetic groups of single fish from different recordings (no interaction by construction; the iso twin
leaves their own recordings out) and time-shifted virtual groups (memgap.shift).

Development on simulated fish only (seeds 801-804): fish with speed relaxation, wall avoidance in a round arena,
persistent (Ornstein-Uhlenbeck) noise, and attraction plus alignment to neighbours (all-range: "bright"; within
0.15 arena radii: "dark"; or one-way: member 1 ignores member 0), seen with a 0.2 s reaction delay. Individuals
differ in speed and noise (+-30%). Development showed:
  * with instantaneous responses, strongly coupled fish lock together and the member twin's share falls (e77's
    limit): pairs 0.006-0.008 at 1 s against 0.014-0.018 at weaker coupling. A 0.2 s reaction delay (fish respond
    to where a neighbour was) restores a share that grows with coupling; the 0.5 s horizon gave the clearest one;
  * nulls (loners, synthetic groups, 9-10 per set): member twin within +-0.004, iso twin within +-0.013 (individual
    differences between single fish), time-shifted groups up to 0.025;
  * all-range coupling (k 0.3) against short-range only: pairs 0.017-0.034 against <= 0.001, triads 0.075-0.111
    against <= 0.002 (Mann-Whitney p <= 0.0005, both twins, seeds 803-804);
  * mutual against one-way coupling at equal strength: 0.116 against 0.055 (pairs, k 1) with the delay, but equal
    (0.022) without it: the VR reciprocity prediction (V2) is the riskiest;
  * memory gap within 0.006 of the measured member share.

Predictions, fixed before any real share was computed (simulations on fresh seeds 805-806):
  S1  Nulls: simulated loners and synthetic groups, |S_member| <= 0.01 and |S_iso| <= 0.02 in every group.
  S2  All-range coupling above short-range only: Mann-Whitney p < 0.05, pairs and triads, both twins.
  S3  Mutual above one-way coupling (pairs, k 1): pooled S_member.
  S4  Memory gap within 0.01 of the measured member share in >= 80% of coupled groups.
  L1  Synthetic groups of real single fish (10 per size and light): |S_member| <= 0.01 and |S_iso| <= 0.02 in >= 80%.
  L2  Time-shifted virtual groups of the real pairs and triads: |S_member| <= 0.03 in >= 80%.
  L3  Bright pairs and bright triads: pooled S_member, 90% lower bound above the largest synthetic-group share.
  L4  The causal test. Dark below bright, Mann-Whitney one-sided p < 0.05 on S_member, in pairs and in triads.
  L5  The same with the iso twin (single fish recorded alone in the same light).
  L6  Memory gap within 0.01 of the measured member share in >= 80% of real recordings.
  L7  Size: pooled dark S_member below half the pooled bright S_member, pairs and triads.
  V1  VR time-shifted virtual groups: |S_member| <= 0.03 in >= 80% of experiments.
  V2  Reciprocity: pooled S_member of C2 (the clone responds) above C3 (the same clone ignores the fish).
  V3  One-way interaction is still interaction: C3 pooled S_member, 90% lower bound > 0.
  V4  Two real fish (2R): pooled S_member, 90% lower bound > 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import intervention_data as data
from experiments.e67_memory_gap import design, s_pred
from experiments.memgap import default_windows, memories, shift

SEED = 80
H = 5
WINDOWS = default_windows(H)
OFFS, F, Y = design(WINDOWS, [0])
BLOCK = 300
FOLDS = 3
SPAN = int(OFFS.max() - OFFS.min())


# ---------------------------------------------------------------- the share

def _feats(x, rows):
    """x (T, ..., 2): window sums at each row; returns (rows, ..., 2, windows) and the target (rows, ..., 2)."""
    w = [sum(x[rows + o] for o in OFFS[F[k] > 0]) for k in range(len(F))]
    return np.stack(w, -1), x[rows]


def _layout(segs, start=0):
    """Rows of each segment and their fold (30 s blocks dealt to folds in turn across segments)."""
    out, b = [], start
    for s in segs:
        rows = np.arange(-int(OFFS.min()), len(s) - int(OFFS.max()))
        blk = (rows - rows[0]) // BLOCK
        out.append((rows, (blk + b) % FOLDS))
        b += int(blk.max()) + 1
    return out


def _purged(rows, fold, k):
    """Training rows for test fold k: rows whose lag footprint stays clear of every test row's footprint."""
    t = int(rows.max()) + 1
    hit = np.zeros(t + 1, int)
    for r in rows[fold == k]:
        hit[max(r + int(OFFS.min()), 0)] += 1
        hit[r + 1] -= 1
    busy = np.cumsum(hit)[:t] > 0
    c = np.concatenate([[0], np.cumsum(busy)])
    lo = np.clip(rows + int(OFFS.min()) - SPAN, 0, t)
    clear = c[np.minimum(rows + SPAN + 1, t)] - c[lo] == 0
    return (fold != k) & clear


def _ols(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def iso_weights(singles):
    """Twin weights from single fish recorded alone: one isotropic law over both coordinates."""
    xs, ys = [], []
    for v in singles:
        rows = np.arange(-int(OFFS.min()), len(v) - int(OFFS.max()))
        f, y = _feats(v[:, 0], rows)
        xs.append(f.reshape(-1, len(F)))
        ys.append(y.reshape(-1))
    return _ols(np.concatenate(xs), np.concatenate(ys))


def shares(segs, iso=None):
    """Cross-fitted S for the summed velocity of segs [(T, N, 2)] against the member twin (and the iso twin if
    weights are given). Returns the shares and per-fold squared errors (for pooling across recordings)."""
    lay = _layout(segs)
    sse = {"group": 0.0, "member": 0.0, "iso": 0.0}
    for k in range(FOLDS):
        gx, gy, mx, my, tx, ty = [], [], [], [], [], []
        for s, (rows, fold) in zip(segs, lay):
            tr, te = _purged(rows, fold, k), fold == k
            fg, yg = _feats(s.sum(1), rows)
            fm, ym = _feats(s, rows)
            gx.append(fg[tr].reshape(-1, len(F)))
            gy.append(yg[tr].reshape(-1))
            mx.append(fm[tr].reshape(-1, len(F)))
            my.append(ym[tr].reshape(-1))
            tx.append(fg[te].reshape(-1, len(F)))
            ty.append(yg[te].reshape(-1))
        gx, gy, mx, my = map(np.concatenate, (gx, gy, mx, my))
        tx, ty = np.concatenate(tx), np.concatenate(ty)
        if not len(ty):
            continue
        bg, bm = _ols(gx, gy), _ols(mx, my)
        sse["group"] += float(((ty - tx @ bg) ** 2).sum())
        sse["member"] += float(((ty - tx @ bm) ** 2).sum())
        if iso is not None:
            sse["iso"] += float(((ty - tx @ iso) ** 2).sum())
    out = {"S_member": 1 - sse["group"] / sse["member"], "sse": sse}
    if iso is not None:
        out["S_iso"] = 1 - sse["group"] / sse["iso"]
    return out


def gaps(segs, singles=None):
    """Memory gaps: member memory from the group's members, and (if given) from single fish recorded alone."""
    o, g = memories(segs, SPAN)
    out = {"gap_member": s_pred(o, g, OFFS, F, Y)}
    if singles is not None:
        oi, _ = memories(singles, SPAN)
        out["gap_iso"] = s_pred(oi, g, OFFS, F, Y)
    return out


def pooled(results, key):
    """Share pooled over recordings: 1 - sum SSE(group) / sum SSE(twin)."""
    return 1 - sum(r["sse"]["group"] for r in results) / sum(r["sse"][key] for r in results)


def boot(results, key, rng, n=400):
    """90% interval of the pooled share, resampling recordings."""
    idx = rng.integers(0, len(results), (n, len(results)))
    v = [pooled([results[i] for i in row], key) for row in idx]
    return [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]


# ---------------------------------------------------------------- simulated fish

SIM = dict(dt=1 / 30, v0=0.25, relax=2.0, sigma=0.35, tau=0.5, wall=0.15, att=1.0, ali=1.5, dark_range=0.15,
           delay=0.2)


def simulate(n, minutes, rule, rng, reps=1, k=1.0, hetero=0.3, p=SIM):
    """Fish in a round arena of radius 1. rule: 'loners', 'bright' (all-range attraction and alignment), 'dark'
    (only within dark_range) or 'oneway' (member 1 ignores member 0; pairs). Returns (reps, T, n, 2) positions at 10 Hz."""
    dt = p["dt"]
    steps = int(minutes * 60 / dt)
    every = int(round(data.DT / dt))
    v0 = p["v0"] * (1 + hetero * rng.uniform(-1, 1, (reps, n, 1)))
    sig = p["sigma"] * (1 + hetero * rng.uniform(-1, 1, (reps, n, 1)))
    r0 = np.sqrt(rng.uniform(0, 0.8, (reps, n, 1)))
    a0 = rng.uniform(0, 2 * np.pi, (reps, n, 1))
    pos = np.concatenate([r0 * np.cos(a0), r0 * np.sin(a0)], -1)
    vel = v0 * rng.normal(size=(reps, n, 2))
    eta = np.zeros((reps, n, 2))
    out = np.empty((reps, steps // every, n, 2))
    decay = np.exp(-dt / p["tau"])
    lag = max(int(round(p["delay"] / dt)), 0)
    hist_p, hist_v = [pos.copy()] * (lag + 1), [vel.copy()] * (lag + 1)
    for t in range(steps):
        hist_p, hist_v = hist_p[1:] + [pos.copy()], hist_v[1:] + [vel.copy()]
        seen_p, seen_v = hist_p[0], hist_v[0]
        sp = np.linalg.norm(vel, axis=-1, keepdims=True) + 1e-9
        acc = -p["relax"] * (sp - v0) * vel / sp
        r = np.linalg.norm(pos, axis=-1, keepdims=True)
        near = np.clip((r - (1 - p["wall"])) / p["wall"], 0, None)
        inward = -pos / (r + 1e-9)
        acc += 6.0 * near * (inward - (vel / sp * inward).sum(-1, keepdims=True) * vel / sp) * v0
        acc -= 4.0 * near * np.clip((vel * -inward).sum(-1, keepdims=True), 0, None) * -inward
        if rule != "loners" and n > 1:
            d = seen_p[:, None, :, :] - pos[:, :, None, :]
            dist = np.linalg.norm(d, axis=-1) + np.eye(n)
            w = np.ones((reps, n, n)) - np.eye(n)
            if rule == "dark":
                w = w * (dist < p["dark_range"])
            if rule == "oneway":
                w[:, 1, 0] = 0.0
            att = (w[..., None] * d / dist[..., None]).sum(2)
            ali = (w[..., None] * (seen_v[:, None] - vel[:, :, None])).sum(2)
            acc += k * (p["att"] * v0 * att + p["ali"] * ali)
        eta = eta * decay + sig * np.sqrt(1 - decay ** 2) * rng.normal(size=eta.shape)
        vel = vel + (acc + eta) * dt
        pos = pos + vel * dt
        rr = np.linalg.norm(pos, axis=-1, keepdims=True)
        pos = np.where(rr > 1, pos / rr * 0.999, pos)
        if t % every == 0 and t // every < out.shape[1]:
            out[:, t // every] = pos
    return out


def synthetic(singles, n, rng, count):
    """Groups of n single fish from different recordings, cut to a common length; returns (members, groups)."""
    out = []
    for _ in range(count):
        pick = rng.choice(len(singles), n, replace=False)
        t = min(len(singles[i]) for i in pick)
        out.append((pick, np.concatenate([singles[i][:t] for i in pick], 1)))
    return out


def _synthetic_rows(singles, n, rng, count, **tags):
    out = []
    for pick, g in synthetic(singles, n, rng, count):
        rest = [s for i, s in enumerate(singles) if i not in pick]
        out.append({**tags, "n": n, "kind": "synthetic", **shares([g], iso_weights(rest)), **gaps([g], rest)})
    return out


# ---------------------------------------------------------------- runs

SIM_SEEDS = (805, 806)
MINUTES = 10.5


def run_sims(seed, quick=False):
    from experiments.intervention_data import velocity
    rng = np.random.default_rng([SEED, seed])
    reps = 5 if quick else 10
    singles = [velocity(p) for p in simulate(1, MINUTES, "loners", rng, reps=reps)]
    iso = iso_weights(singles)
    rows = []
    sets = [(n, rule, k) for n in (2, 3) for rule, k in (("bright", 0.3), ("dark", 0.3), ("loners", 1.0))]
    sets += [(2, "bright", 1.0), (2, "oneway", 1.0)]
    for n, rule, k in sets:
        for p in simulate(n, MINUTES, rule, rng, reps=reps, k=k):
            v = velocity(p)
            rows.append({"seed": seed, "n": n, "rule": rule, "k": k, "kind": rule, **shares([v], iso),
                         **gaps([v], singles)})
    for n in (2, 3):
        rows += _synthetic_rows(singles, n, rng, reps, seed=seed, rule="synthetic", k=0.0)
    return rows


def run_light(quick=False):
    from experiments.intervention_data import velocity
    rng = np.random.default_rng([SEED, 1])
    rows = []
    for light in data.LIGHTS:
        singles = [velocity(p) for _, p in data.liu("Singles", light)]
        iso = iso_weights(singles)
        for size in ("Pairs", "Triads"):
            recs = data.liu(size, light)[: 2 if quick else None]
            for name, p in recs:
                v = velocity(p)
                rows.append({"light": light, "n": data.SIZES[size], "rec": name, "kind": "real", **shares([v], iso),
                             **gaps([v], singles), "shift": shares(shift([v], rng))["S_member"]})
            rows += _synthetic_rows(singles, data.SIZES[size], rng, 2 if quick else 10, light=light)
    return rows


def run_vr(quick=False):
    from experiments.intervention_data import velocity
    rng = np.random.default_rng([SEED, 2])
    rows = []
    for cond in data.VR_CONDITIONS:
        exps = data.vr(cond)
        for e in list(exps)[: 2 if quick else None]:
            vs = [velocity(s) for s in exps[e]]
            rows.append({"condition": cond, "exp": e, "segments": len(vs), "seconds": sum(len(v) for v in vs) * data.DT,
                         **shares(vs), **gaps(vs), "shift": shares(shift(vs, rng))["S_member"]})
    return rows


def summarise(res):
    """Pooled shares with 90% intervals (resampling recordings / experiments)."""
    rng = np.random.default_rng([SEED, 3])
    out = {}
    for light in data.LIGHTS:
        for n in (2, 3):
            rs = [r for r in res["light"] if r["light"] == light and r["n"] == n and r["kind"] == "real"]
            out[f"{light}{n}"] = {k: {"S": pooled(rs, k), "ci": boot(rs, k, rng)} for k in ("member", "iso")}
    for cond in data.VR_CONDITIONS:
        rs = [r for r in res["vr"] if r["condition"] == cond]
        out[cond] = {"member": {"S": pooled(rs, "member"), "ci": boot(rs, "member", rng)}}
    c2 = [r for r in res["vr"] if r["condition"] == "C2"]
    c3 = [r for r in res["vr"] if r["condition"] == "C3"]
    i2, i3 = rng.integers(0, len(c2), (400, len(c2))), rng.integers(0, len(c3), (400, len(c3)))
    d = [pooled([c2[i] for i in a], "member") - pooled([c3[i] for i in b], "member") for a, b in zip(i2, i3)]
    out["C2-C3"] = {"S": out["C2"]["member"]["S"] - out["C3"]["member"]["S"],
                    "ci": [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))]}
    return out


# ---------------------------------------------------------------- predictions

def _mw(a, b):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(a, b, alternative="greater").pvalue)


def predictions(res):
    sim, light, vr, summ = res["sim"], res["light"], res["vr"], res["summary"]
    col = lambda rs, k: np.array([r[k] for r in rs])  # noqa: E731
    out = []

    nulls = [r for r in sim if r["kind"] in ("loners", "synthetic")]
    m1, i1 = np.abs(col(nulls, "S_member")).max(), np.abs(col(nulls, "S_iso")).max()
    out.append(("S1", "simulated nulls (loners, synthetic groups): |S_member| <= 0.01 and |S_iso| <= 0.02, every group",
                m1 <= 0.01 and i1 <= 0.02, f"max |S_member| {m1:.3f}, max |S_iso| {i1:.3f} ({len(nulls)} groups)"))
    ps = []
    for n in (2, 3):
        for k in ("S_member", "S_iso"):
            b = [r for r in sim if r["n"] == n and r["rule"] == "bright" and r["k"] == 0.3]
            d = [r for r in sim if r["n"] == n and r["rule"] == "dark"]
            ps.append(_mw(col(b, k), col(d, k)))
    out.append(("S2", "simulated all-range coupling above short-range only, Mann-Whitney p < 0.05 (pairs, triads; both twins)",
                max(ps) < 0.05, "p " + ", ".join(f"{p:.4f}" for p in ps)))
    mu = pooled([r for r in sim if r["rule"] == "bright" and r["n"] == 2 and r["k"] == 1.0], "member")
    ow = pooled([r for r in sim if r["rule"] == "oneway"], "member")
    out.append(("S3", "simulated mutual above one-way coupling (pairs, k 1), pooled S_member", mu > ow,
                f"mutual {mu:+.3f}, one-way {ow:+.3f}"))
    cpl = [r for r in sim if r["rule"] in ("bright", "oneway")]
    s4 = (np.abs(col(cpl, "gap_member") - col(cpl, "S_member")) <= 0.01).mean()
    out.append(("S4", "simulated memory gap within 0.01 of the measured member share in >= 80% of coupled groups",
                s4 >= 0.8, f"{s4:.0%} of {len(cpl)}"))

    syn = [r for r in light if r["kind"] == "synthetic"]
    l1m = (np.abs(col(syn, "S_member")) <= 0.01).mean()
    l1i = (np.abs(col(syn, "S_iso")) <= 0.02).mean()
    out.append(("L1", "synthetic groups of real single fish: |S_member| <= 0.01 and |S_iso| <= 0.02 in >= 80%",
                l1m >= 0.8 and l1i >= 0.8, f"member {l1m:.0%}, iso {l1i:.0%} of {len(syn)}; max member "
                f"{np.abs(col(syn, 'S_member')).max():.3f}, max iso {np.abs(col(syn, 'S_iso')).max():.3f}"))
    real = [r for r in light if r["kind"] == "real"]
    l2 = (np.abs(col(real, "shift")) <= 0.03).mean()
    out.append(("L2", "time-shifted virtual groups of real pairs and triads: |S_member| <= 0.03 in >= 80%", l2 >= 0.8,
                f"{l2:.0%} of {len(real)}; range {col(real, 'shift').min():+.3f} to {col(real, 'shift').max():+.3f}"))
    smax = col(syn, "S_member").max()
    l3 = [summ[f"Bright{n}"]["member"]["ci"][0] for n in (2, 3)]
    out.append(("L3", "bright pairs and triads: pooled S_member, 90% lower bound above the largest synthetic share",
                min(l3) > smax, f"lower bounds {l3[0]:+.3f} (pairs), {l3[1]:+.3f} (triads); synthetic max {smax:+.3f}"))
    for key, tw, label in (("L4", "S_member", "member twin"), ("L5", "S_iso", "iso twin (single fish alone)")):
        p = []
        for n in (2, 3):
            b = [r for r in real if r["n"] == n and r["light"] == "Bright"]
            d = [r for r in real if r["n"] == n and r["light"] == "Dark"]
            p.append(_mw(col(b, tw), col(d, tw)))
        out.append((key, f"causal test: dark below bright, Mann-Whitney one-sided p < 0.05, pairs and triads ({label})",
                    max(p) < 0.05, f"p {p[0]:.4f} (pairs), {p[1]:.4f} (triads)"))
    l6 = (np.abs(col(real, "gap_member") - col(real, "S_member")) <= 0.01).mean()
    out.append(("L6", "memory gap within 0.01 of the measured member share in >= 80% of real recordings", l6 >= 0.8,
                f"{l6:.0%} of {len(real)}"))
    l7 = [(summ[f"Dark{n}"]["member"]["S"], summ[f"Bright{n}"]["member"]["S"]) for n in (2, 3)]
    out.append(("L7", "size: pooled dark S_member below half of bright, pairs and triads",
                all(d < 0.5 * b for d, b in l7), "; ".join(f"N{n}: dark {d:+.3f}, bright {b:+.3f}"
                                                            for n, (d, b) in zip((2, 3), l7))))

    v1 = (np.abs(col(vr, "shift")) <= 0.03).mean()
    out.append(("V1", "VR time-shifted virtual groups: |S_member| <= 0.03 in >= 80% of experiments", v1 >= 0.8,
                f"{v1:.0%} of {len(vr)}; range {col(vr, 'shift').min():+.3f} to {col(vr, 'shift').max():+.3f}"))
    c2, c3 = summ["C2"]["member"]["S"], summ["C3"]["member"]["S"]
    out.append(("V2", "reciprocity: pooled S_member, C2 (clone responds) above C3 (same clone ignores the fish)",
                c2 > c3, f"C2 {c2:+.3f}, C3 {c3:+.3f}, difference 90% {np.round(summ['C2-C3']['ci'], 3).tolist()}"))
    lo3 = summ["C3"]["member"]["ci"][0]
    out.append(("V3", "one-way interaction is detected: C3 pooled S_member, 90% lower bound > 0", lo3 > 0,
                f"C3 {c3:+.3f} [{lo3:+.3f}, {summ['C3']['member']['ci'][1]:+.3f}]"))
    lo2r = summ["2R"]["member"]["ci"][0]
    out.append(("V4", "two real fish: pooled S_member, 90% lower bound > 0", lo2r > 0,
                f"2R {summ['2R']['member']['S']:+.3f} [{lo2r:+.3f}, {summ['2R']['member']['ci'][1]:+.3f}]"))
    return out


def analyze(res) -> str:
    col = lambda rs, k: np.array([r[k] for r in rs])  # noqa: E731
    lines = ["Experiment 80: switching interaction off", "", "  Simulated fish (fresh seeds), medians:",
             "  N  rule       k     groups  S_member  S_iso   gap_member  gap_iso"]
    sim = res["sim"]
    for key in sorted({(r["n"], r["rule"], r["k"]) for r in sim}):
        rs = [r for r in sim if (r["n"], r["rule"], r["k"]) == key]
        lines.append(f"  {key[0]}  {key[1]:10} {key[2]:<4}  {len(rs):<6d}  " + "  ".join(
            f"{np.median(col(rs, k)):+.3f}  " for k in ("S_member", "S_iso", "gap_member", "gap_iso")))
    lines += ["", "  Real zebrafish, light and dark (one row per recording):",
              "  light   N  rec             S_member  S_iso    gap_member gap_iso  shifted"]
    for r in res["light"]:
        if r["kind"] == "real":
            lines.append(f"  {r['light']:7} {r['n']}  {r['rec']:14}  {r['S_member']:+.3f}    {r['S_iso']:+.3f}   "
                         f"{r['gap_member']:+.3f}     {r['gap_iso']:+.3f}   {r['shift']:+.3f}")
    syn = [r for r in res["light"] if r["kind"] == "synthetic"]
    lines.append(f"  synthetic groups of single fish ({len(syn)}): S_member {col(syn, 'S_member').min():+.3f} to "
                 f"{col(syn, 'S_member').max():+.3f}, S_iso {col(syn, 'S_iso').min():+.3f} to {col(syn, 'S_iso').max():+.3f}")
    lines += ["", "  Pooled shares [90%]:"]
    for key, v in res["summary"].items():
        if key == "C2-C3":
            lines.append(f"    C2 - C3          {v['S']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
            continue
        lines.append(f"    {key:8} " + "  ".join(f"{t} {s['S']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]"
                                               for t, s in v.items()))
    lines += ["", "  VR experiments:", "  cond  exp  segments  seconds  S_member  gap_member  shifted"]
    for r in res["vr"]:
        lines.append(f"  {r['condition']:4}  {r['exp']:<3d}  {r['segments']:<8d}  {r['seconds']:7.0f}  "
                     f"{r['S_member']:+.3f}    {r['gap_member']:+.3f}      {r['shift']:+.3f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    res = {"sim": [r for s in SIM_SEEDS[: 1 if quick else None] for r in run_sims(s, quick)],
           "light": run_light(quick), "vr": run_vr(quick)}
    res["summary"] = summarise(res)
    (out_dir / "e80_intervention_raw.json").write_text(json.dumps(res, default=float))
    text = analyze(res)
    from experiments.e80_figures import make_figures
    make_figures(res, out_dir / "e80_figures", text)
    return text


if __name__ == "__main__":
    main()
