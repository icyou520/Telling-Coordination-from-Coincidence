"""Experiment 86: the Ising magnet. Does the memory gap behave like an order parameter at a phase transition?

The 2D Ising model on an L x L periodic lattice (coupling J = 1, no field) with heat-bath dynamics, updated in
checkerboard half-sweeps; one time step is one sweep. Its critical temperature is known exactly,
Tc = 2 / ln(1 + sqrt 2) = 2.269. Members are the L^2 spins, the group is the magnetization (their sum), and the
interaction share is the memory gap (e67; twinshare.engine.s_pred, windows 1, 5, 22 sweeps): how much better the
magnetization's own history forecasts it than the sum of each spin's forecast from its own history.
    high T   spins nearly independent: the share should vanish;
    low T    the lattice is ordered and each spin's own past already says which way it points;
    near Tc  large, slow collective fluctuations that no single spin's past carries.
Controls: J = 0 (independent spins with the same update rule) and a virtual lattice (each spin's record circularly
shifted by its own offset).

Development (L = 8 and 16, seed 0, 2,000 + 20,000 sweeps, T 1.6-3.4 in steps of 0.1 plus 2.15, 2.25, 2.35, 4, 6, 10):
  * the share is small in the ordered phase (0.06-0.10 at T 1.6), rises steeply through Tc, has a broad maximum
    above it (L 8: 0.582 at T 2.5; L 16: 0.683 at T 2.5; the susceptibility peaks at 2.6 and 2.4), and falls
    slowly at high T (0.35-0.37 at T 4, 0.16-0.17 at 6, 0.05-0.06 at 10): it vanishes at both extremes;
  * far above Tc it hardly depends on size (L 8 against 16: within 0.014 at T >= 3.4), as short-range physics should;
  * controls: virtual lattices <= 0.010, J = 0 <= 0.0004 at every T;
  * near Tc single runs are noisy (L 16: 0.584 at 2.10, 0.521 at 2.15), since the magnetization relaxes slowly:
    the final run uses 5,000 + 40,000 sweeps.

Pre-registered predictions (final run: L = 16, 32, 64, seed 1, 5,000 + 40,000 sweeps, the development grid):
  I1  Controls: |S| <= 0.02 for virtual lattices and J = 0, every L and T.
  I2  Vanishing at both extremes: S < 0.15 at T 1.6 and S < 0.10 at T 10, every L.
  I3  A peak in the critical region: for every L the largest S is above 0.5 and lies at 2.2 <= T <= 2.9.
  I4  The peak grows with size: max S at L 64 > L 32 > L 16.
  I5  Size-independent far above Tc: |S(L 32) - S(L 16)| and |S(L 64) - S(L 16)| <= 0.03 at T = 3.4, 4, 6, 10.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twinshare" / "src"))
from twinshare import engine as E  # noqa: E402

SEED = 86
TC = 2 / np.log(1 + np.sqrt(2))
DESIGN = E.Design(windows=(1, 5, 22))
SWEEPS = 20000
BURN = 2000
TEMPS = tuple(np.round(np.concatenate([np.arange(1.6, 3.41, 0.1), [2.15, 2.25, 2.35]]), 3))


def simulate(L, T, sweeps, rng, J=1.0, burn=BURN):
    """Spins (sweeps, L*L) as int8. Ordered start below Tc, random above."""
    s = np.ones((L, L), np.int8) if T < TC else rng.choice(np.array([-1, 1], np.int8), (L, L))
    ii, jj = np.indices((L, L))
    masks = [(ii + jj) % 2 == p for p in (0, 1)]
    p_up = 1 / (1 + np.exp(-2 * J * np.arange(-4, 5) / T))
    out = np.empty((sweeps, L * L), np.int8)
    for t in range(burn + sweeps):
        for m in masks:
            nb = (np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)).astype(np.int64)
            new = np.where(rng.random((L, L)) < p_up[nb + 4], 1, -1).astype(np.int8)
            s = np.where(m, new, s)
        if t >= burn:
            out[t - burn] = s.ravel()
    return out


def memories(x, maxlag, chunk=256):
    """E.memories via FFT: members' mean autocovariance and the sum's autocovariance (biased), lags 0..maxlag."""
    t, n = x.shape
    nfft = 1 << int(np.ceil(np.log2(2 * t)))
    o = np.zeros(maxlag + 1)
    total = np.zeros(t)
    for a in range(0, n, chunk):
        u = x[:, a:a + chunk].astype(np.float64)
        u -= u.mean(0)
        total += u.sum(1)
        f = np.fft.rfft(u, nfft, axis=0)
        o += np.fft.irfft((f * f.conj()).real.sum(1), nfft)[: maxlag + 1]
    f = np.fft.rfft(total, nfft)
    g = np.fft.irfft((f * f.conj()).real, nfft)[: maxlag + 1]
    return o / (n * t), g / t


def gap(x):
    return E.s_pred(*memories(x, DESIGN.lead), DESIGN)


def shift(x, rng):
    t, n = x.shape
    m = max(t // 20, 1)
    off = rng.integers(m, t - m + 1, n)
    out = np.empty_like(x)
    for i in range(n):
        out[:, i] = np.roll(x[:, i], -off[i])
    return out


def run_point(L, T, seed, sweeps=SWEEPS, controls=True):
    rng = np.random.default_rng([SEED, seed, L, int(round(T * 1000))])
    x = simulate(L, T, sweeps, rng)
    m = x.sum(1, dtype=np.int64) / x.shape[1]
    row = {"L": L, "T": float(T), "seed": seed, "S": gap(x), "abs_m": float(np.abs(m).mean()),
           "chi": float(x.shape[1] * (np.mean(m ** 2) - np.abs(m).mean() ** 2) / T),
           "rho_spin": float(np.mean([np.corrcoef(x[:-1, i], x[1:, i])[0, 1] for i in range(0, x.shape[1], 7)
                                      if x[:, i].std() > 0]))}
    if controls:
        row["S_virtual"] = gap(shift(x, rng))
        row["S_free"] = gap(simulate(L, T, sweeps, rng, J=0.0))
    return row


def scan(L, seed, temps=TEMPS, sweeps=SWEEPS, controls=True):
    return [run_point(L, T, seed, sweeps, controls) for T in sorted(temps)]


SIZES = (16, 32, 64)
FINAL = dict(sweeps=40000, burn=5000)
HIGH = (3.4, 4.0, 6.0, 10.0)
GRID = TEMPS + (4.0, 6.0, 10.0)


def describe(rows):
    lines = ["    T      S        S virtual  S J=0     |m|     chi      spin lag-1"]
    for r in rows:
        lines.append(f"    {r['T']:5.2f}  {r['S']:+.4f}  {r.get('S_virtual', np.nan):+.4f}    "
                     f"{r.get('S_free', np.nan):+.4f}   {r['abs_m']:.3f}   {r['chi']:7.2f}  {r['rho_spin']:+.3f}")
    return lines


def predictions(res):
    rows = {int(k): v for k, v in res["scans"].items()}
    at = lambda L, T: next(r["S"] for r in rows[L] if abs(r["T"] - T) < 1e-9)  # noqa: E731
    out = []
    ctrl = max(abs(r[k]) for v in rows.values() for r in v for k in ("S_virtual", "S_free"))
    out.append(("I1", "controls |S| <= 0.02 (virtual lattices, J = 0)", ctrl <= 0.02, f"max {ctrl:.4f}"))
    lo, hi = [at(L, 1.6) for L in rows], [at(L, 10.0) for L in rows]
    out.append(("I2", "vanishing at both extremes: S < 0.15 at T 1.6, < 0.10 at T 10",
                max(lo) < 0.15 and max(hi) < 0.10,
                f"T 1.6: {', '.join(f'{s:.3f}' for s in lo)}; T 10: {', '.join(f'{s:.3f}' for s in hi)}"))
    peaks = {L: max(v, key=lambda r: r["S"]) for L, v in rows.items()}
    out.append(("I3", "peak > 0.5 at 2.2 <= T <= 2.9, every L",
                all(p["S"] > 0.5 and 2.2 <= p["T"] <= 2.9 for p in peaks.values()),
                "; ".join(f"L {L}: {p['S']:.3f} at T {p['T']:.2f}" for L, p in peaks.items())))
    sizes = sorted(rows)
    h = [peaks[L]["S"] for L in sizes]
    out.append(("I4", "peak grows with size: L 64 > L 32 > L 16", all(np.diff(h) > 0),
                ", ".join(f"L {L}: {s:.3f}" for L, s in zip(sizes, h))))
    d = [abs(at(L, T) - at(sizes[0], T)) for L in sizes[1:] for T in HIGH]
    out.append(("I5", "far above Tc, within 0.03 of L 16 (T 3.4, 4, 6, 10)", max(d) <= 0.03, f"max {max(d):.4f}"))
    return out


def analyze(res):
    lines = ["Experiment 86: the Ising magnet. The memory gap across the phase transition (Tc = 2.269)", ""]
    for L, rows in res["scans"].items():
        lines += [f"  L = {L} ({int(L) ** 2} spins), {FINAL['burn']:,} + {FINAL['sweeps']:,} sweeps"] + describe(rows) + [""]
    lines.append("  Pre-registered predictions:")
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    sizes, grid = ((8, 10, 12), GRID[::3] + HIGH + (1.6,)) if quick else (SIZES, GRID)
    kw = dict(sweeps=4000, burn=500) if quick else FINAL
    res = {"scans": {}}
    for L in sizes:
        rows = []
        for T in sorted(set(grid)):
            rng = np.random.default_rng([SEED, 1, L, int(round(T * 1000))])
            x = simulate(L, T, kw["sweeps"], rng, burn=kw["burn"])
            m = x.sum(1, dtype=np.int64) / x.shape[1]
            rows.append({"L": L, "T": float(T), "S": gap(x), "abs_m": float(np.abs(m).mean()),
                         "chi": float(x.shape[1] * (np.mean(m ** 2) - np.abs(m).mean() ** 2) / T),
                         "rho_spin": float(np.mean([np.corrcoef(x[:-1, i], x[1:, i])[0, 1]
                                                    for i in range(0, x.shape[1], 7) if x[:, i].std() > 0])),
                         "S_virtual": gap(shift(x, rng)),
                         "S_free": gap(simulate(L, T, kw["sweeps"], rng, J=0.0, burn=kw["burn"]))})
            del x
        res["scans"][str(L)] = rows
    (out_dir / "e86_ising_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e86_figures import make_figures
    make_figures(res, out_dir / "e86_figures", text)
    return text


if __name__ == "__main__":
    main()
