"""Memory-gap screening tool: predict a group's interaction share from two memories, with no twin to fit.

S_pred is the cost of forecasting the group (the sum of its members) with its members' memory instead of its own
(e67). It matched the measured interaction share in simulations, brain areas and markets (e67, e69), and
separated real from virtual fish schools (e68). Use it to screen data before a full study:

  * remove recorded causes first (stimulus responses, seasonality, trends), or they count as shared memory;
  * the virtual baseline (members circularly shifted against each other) should be about 0;
  * S_pred measures coordination beyond members' own histories; it cannot say whether it comes from members
    interacting or from an unrecorded common input with the same memories (e67, P2);
  * for a nonlinear group variable (for example a school's polarization) the size of S_pred is not the size of
    that variable's share (e68), only whether there is coordination.

Command line, for a CSV with one row per time step and one column per member:

    uv run python -m experiments.memgap data.csv --horizon 1 [--log]
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.e67_memory_gap import design, s_one, s_pred


def default_windows(horizon):
    h = max(1, int(horizon))
    return ((h, h + 1), (h + 1, 2 * h + 1), (2 * h + 1, 3 * h + 2))


def memories(segments, maxlag):
    """segments: list of arrays (T, N) or (T, N, d) for vector members. Lagged products never cross segments.
    Returns member-mean autocovariance o(k) and group autocovariance g(k), k = 0..maxlag. Biased estimator (every
    lag divided by the total length), so both stay positive definite even for very smooth series."""
    segs = [s if s.ndim == 3 else s[..., None] for s in segments if len(s) > maxlag + 1]
    mean = np.concatenate(segs).mean(0, keepdims=True)
    so, sg = np.zeros(maxlag + 1), np.zeros(maxlag + 1)
    for s in segs:
        u = s - mean
        v = u.sum(1)
        t = len(u)
        for k in range(maxlag + 1):
            so[k] += (u[: t - k] * u[k:]).sum(-1).mean(1).sum()
            sg[k] += (v[: t - k] * v[k:]).sum(-1).sum()
    total = sum(len(s) for s in segs)
    return so / total, sg / total


def shift(segments, rng):
    """Virtual baseline: each member circularly shifted by its own offset within each segment."""
    out = []
    for s in segments:
        t, n = s.shape[:2]
        idx = (np.arange(t)[:, None] + rng.integers(t // 4, 3 * t // 4 + 1, n)[None]) % t
        out.append(s[idx, np.arange(n)[None]])
    return out


def permute(segments, rng):
    """Virtual baseline for trials (equal-length segments): each member's trials shuffled independently."""
    x = np.stack(segments)
    idx = np.stack([rng.permutation(len(x)) for _ in range(x.shape[2])], 1)
    return list(x[idx, :, np.arange(x.shape[2])[None]].swapaxes(1, 2))


def screen(segments, windows=None, horizon=1, target=(0,), shifts=5, baseline="shift", seed=0):
    """windows: lag ranges [a, b) before the target offsets (e67 convention). The virtual baseline ("shift" in
    time, or "permute" trials across members) is the median over `shifts` draws; it is the floor below which
    S_pred says nothing (slow drifts in short or long-memory series create spurious cross-covariance that it
    picks up; within-trial shifts keep trial-wide shared fluctuations, so use "permute" for trials)."""
    shuffle = permute if baseline == "permute" else shift
    windows = windows or default_windows(horizon)
    offs, f, y = design(windows, list(target))
    maxlag = int(offs.max() - offs.min())
    o, g = memories(segments, maxlag)
    rng = np.random.default_rng(seed)
    virtual = [s_pred(*memories(shuffle(segments, rng), maxlag), offs, f, y) for _ in range(shifts)]
    n = segments[0].shape[1]
    return {"members": int(n), "steps": int(sum(len(s) for s in segments)), "S_pred": s_pred(o, g, offs, f, y),
            "S_1": s_one(o, g), "rho_m": float(o[1] / o[0]), "rho_g": float(g[1] / g[0]),
            "virtual": float(np.median(virtual))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--horizon", type=int, default=1, help="forecast horizon in rows")
    ap.add_argument("--log", action="store_true", help="use log(|x| + 0.05), for heavy-tailed series")
    a = ap.parse_args()
    x = np.genfromtxt(a.csv, delimiter=",", skip_header=1)
    x = x[np.isfinite(x).all(1)]
    if a.log:
        x = np.log(np.abs(x) + 0.05)
    r = screen([x], horizon=a.horizon)
    print(f"members {r['members']}, steps {r['steps']}")
    print(f"S_pred {r['S_pred']:+.4f}  (virtual floor {r['virtual']:+.4f}; below or near it, the screen is silent)")
    print(f"member memory {r['rho_m']:.3f}, group memory {r['rho_g']:.3f}, one-lag equation {r['S_1']:+.4f}")


if __name__ == "__main__":
    main()
