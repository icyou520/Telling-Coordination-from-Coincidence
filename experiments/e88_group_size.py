"""Experiment 88: group size from 2 to 10,000. Does the share scale, saturate or vanish?

One alignment rule, applied at every size: each agent's 2D velocity
    v_i(t+1) = a [(1 - w) v_i(t) + w mean_{j in nb(i)} v_j(t)] + sigma eps_i(t),   a = 0.9, w = 0.5, sigma = 1,
and positions move with the velocities in a periodic square at density 1 (side sqrt(N)). Neighbourhoods:
    mean-field   every other agent;
    topological  the 7 nearest (as measured in starling flocks);
    metric       every agent within distance 1.5 (about 7 on average; an agent alone keeps its own velocity).
The group is the summed velocity; the share is the memory gap (twinshare.engine.s_pred, windows 1, 5, 22 steps),
from members' mean autocovariance (both components) and the group's.

Exact theory for mean-field. The sum V follows V' = a V + noise whatever w is, and each member is V/N plus a
deviation with AR coefficient b = a(1 - w) - a w / (N - 1), so (per unit noise)
    g(k) = N a^k / (1 - a^2),   o(k) = a^k / (N (1 - a^2)) + (N - 1)/N * b^k / (1 - b^2),
and S(N) = s_pred(o, g): 0.142 at N = 2 (where b = 0), 0.301 at 10, 0.5155 at 10,000, limit 0.5159.

Development (seed 0, N = 2-1,000, 10,000 steps):
  * mean-field: measured within 0.0073 of theory at every N;
  * the share rises from 0.14 (N 2) to about N 30, then saturates: mean-field 0.50-0.52 (N 300-1,000),
    topological 0.39-0.41, metric 0.38-0.39 (N 100-1,000);
  * the group's lag-1 memory is 0.90 (= a) at every size and rule; members' memory falls with N and settles
    (mean-field 0.45, local rules 0.52): local neighbours share more of an agent's own memory;
  * N 2 and 3: all rules coincide (every agent sees every other); virtual groups <= 0.0003.

Pre-registered predictions (seed 1, N = 2-10,000):
  G1  Mean-field matches theory: |S - S_theory| <= 0.015 at every N.
  G2  Virtual groups: |S| <= 0.005 everywhere.
  G3  Saturation: for every rule, S at N 1,000, 3,000 and 10,000 within 0.03 of each other.
  G4  Rise in small groups: S(2) < S(3) < S(5) < S(10) < S(30) for every rule.
  G5  N 2 and 3: the three rules within 0.01 of each other.
  G6  Local interaction saturates lower: topological and metric S at N 10,000 below mean-field's by > 0.05.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twinshare" / "src"))
from twinshare import engine as E  # noqa: E402

SEED = 88
A, W, SIGMA = 0.9, 0.5, 1.0
DESIGN = E.Design(windows=(1, 5, 22))
RULES = ("mean-field", "topological", "metric")
SIZES = (2, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000)
STEPS, BURN = 10000, 500
KEEP = 1000
K_NEAR, RADIUS = 7, 1.5


def theory(n, a=A, w=W, maxlag=DESIGN.lead):
    k = np.arange(maxlag + 1)
    b = a * (1 - w) - a * w / (n - 1)
    g = n * a ** k / (1 - a ** 2)
    o = a ** k / (n * (1 - a ** 2)) + (n - 1) / n * b ** k / (1 - b ** 2)
    return E.s_pred(o, g, DESIGN)


def neighbour_mean(rule, pos, v, side):
    n = len(v)
    if rule == "mean-field":
        return (v.sum(0) - v) / (n - 1)
    tree = cKDTree(pos, boxsize=side)
    if rule == "topological":
        k = min(K_NEAR, n - 1)
        _, idx = tree.query(pos, k + 1)
        return v[idx[:, 1:]].mean(1)
    d, idx = tree.query(pos, min(n, 40), distance_upper_bound=RADIUS)
    ok = np.isfinite(d[:, 1:])
    idx = np.where(ok, idx[:, 1:], 0)
    cnt = ok.sum(1)
    s = (v[idx] * ok[..., None]).sum(1)
    return np.where(cnt[:, None] > 0, s / np.maximum(cnt, 1)[:, None], v)


def simulate(rule, n, rng, steps=STEPS, burn=BURN, keep=KEEP):
    """Returns velocities of up to `keep` agents (steps, keep, 2) and the summed velocity (steps, 2)."""
    side = np.sqrt(n)
    pos = rng.random((n, 2)) * side
    v = rng.normal(0, SIGMA / np.sqrt(1 - A ** 2), (n, 2))
    kept = rng.choice(n, min(n, keep), replace=False)
    out = np.empty((steps, len(kept), 2), np.float32)
    total = np.empty((steps, 2))
    for t in range(burn + steps):
        v = A * ((1 - W) * v + W * neighbour_mean(rule, pos, v, side)) + SIGMA * rng.normal(size=(n, 2))
        pos = (pos + 0.1 * v) % side
        if t >= burn:
            out[t - burn] = v[kept]
            total[t - burn] = v.sum(0)
    return out, total


def autocov(x, maxlag):
    """Biased autocovariance summed over columns, lags 0..maxlag (x: (T, m))."""
    t = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * t)))
    u = x - x.mean(0)
    f = np.fft.rfft(u, nfft, axis=0)
    return np.fft.irfft((f * f.conj()).real.sum(1), nfft)[: maxlag + 1] / t


def measure(rule, n, seed, steps=STEPS):
    rng = np.random.default_rng([SEED, seed, RULES.index(rule), n])
    vel, total = simulate(rule, n, rng, steps)
    lag = DESIGN.lead
    o = autocov(vel.reshape(len(vel), -1).astype(np.float64), lag) / vel.shape[1]
    g = autocov(total, lag)
    out = {"rule": rule, "N": n, "seed": seed, "S": E.s_pred(o, g, DESIGN),
           "rho_member": float(o[1] / o[0]), "rho_group": float(g[1] / g[0])}
    if rule == "mean-field":
        out["theory"] = theory(n)
    # virtual group: each kept member shifted in time (sum rebuilt from the kept members, scaled to N)
    m = max(len(vel) // 20, 1)
    off = rng.integers(m, len(vel) - m + 1, vel.shape[1])
    sh = np.stack([np.roll(vel[:, i], -off[i], 0) for i in range(vel.shape[1])], 1).astype(np.float64)
    out["S_virtual"] = E.s_pred(autocov(sh.reshape(len(sh), -1), lag) / sh.shape[1], autocov(sh.sum(1), lag), DESIGN)
    return out


def describe(rows):
    lines = ["    rule         N       S        theory   virtual   member lag-1  group lag-1"]
    for r in rows:
        lines.append(f"    {r['rule']:11s}  {r['N']:<6d}  {r['S']:+.4f}  {r.get('theory', np.nan):+.4f}  "
                     f"{r['S_virtual']:+.4f}   {r['rho_member']:+.3f}        {r['rho_group']:+.3f}")
    return lines


def predictions(rows):
    at = {(r["rule"], r["N"]): r["S"] for r in rows}
    sizes = sorted({r["N"] for r in rows})
    out = []
    mf = [r for r in rows if r["rule"] == "mean-field"]
    dev = max(abs(r["S"] - r["theory"]) for r in mf)
    out.append(("G1", "mean-field within 0.015 of exact theory, every N", dev <= 0.015, f"max |diff| {dev:.4f}"))
    v = max(abs(r["S_virtual"]) for r in rows)
    out.append(("G2", "virtual groups |S| <= 0.005", v <= 0.005, f"max {v:.4f}"))
    big = [n for n in sizes if n >= 1000]
    spread = {rule: max(at[rule, n] for n in big) - min(at[rule, n] for n in big) for rule in RULES}
    out.append(("G3", "saturation: S at N >= 1,000 within 0.03, every rule", max(spread.values()) <= 0.03,
                ", ".join(f"{k} {s:.4f}" for k, s in spread.items())))
    small = [n for n in sizes if n <= 30]
    rise = {rule: all(np.diff([at[rule, n] for n in small]) > 0) for rule in RULES}
    out.append(("G4", "S rises from N 2 to 30, every rule", all(rise.values()),
                "; ".join(f"{k}: " + " ".join(f"{at[k, n]:.3f}" for n in small) for k in RULES)))
    tiny = [max(at[r, n] for r in RULES) - min(at[r, n] for r in RULES) for n in (2, 3)]
    out.append(("G5", "N 2 and 3: rules within 0.01", max(tiny) <= 0.01, ", ".join(f"{d:.4f}" for d in tiny)))
    n = max(sizes)
    gaps = [at["mean-field", n] - at[r, n] for r in RULES[1:]]
    out.append(("G6", f"local rules below mean-field at N {n:,} by > 0.05", min(gaps) > 0.05,
                ", ".join(f"{g:.4f}" for g in gaps)))
    return out


def analyze(rows):
    lines = ["Experiment 88: group size from 2 to 10,000 under one alignment rule", ""] + describe(rows)
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(rows):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    sizes, seed, steps = ((2, 3, 5, 10, 30, 1000), 101, 2000) if quick else (SIZES, 1, STEPS)
    rows = [measure(r, n, seed, steps) for r in RULES for n in sizes]
    (out_dir / "e88_group_size_raw.json").write_text(json.dumps(rows))
    text = analyze(rows)
    from experiments.e88_figures import make_figures
    make_figures(rows, out_dir / "e88_figures", text)
    return text


if __name__ == "__main__":
    main()
