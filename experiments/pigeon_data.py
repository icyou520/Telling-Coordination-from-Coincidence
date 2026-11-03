"""Loader for the pigeon free-flight GPS trajectories of Nagy et al. (2013,
PNAS 110, 13049; data: https://coordinatedflock.elte.hu/, free to use with
citation). Unzip trajectoriesD2.zip into data/pigeons/D2/.

Each flight is returned at the native GPS rate of 0.2 s (every second 10 Hz
row), restricted to the samples where every bird flies faster than 5 m/s.
Velocities are the dataset's own (spline) derivatives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "pigeons" / "D2" / "flight"
DT = 0.2
MIN_FLY_SPEED = 5.0
FLOCKS = ("A", "B", "C")


def flights(flock: str, n_birds: int = 10) -> list[str]:
    return sorted(d.name for d in DATA.glob(f"ff{flock}*")
                  if sum(1 for _ in d.iterdir()) == n_birds)


def load(flight: str, stride: int = 2) -> dict:
    """Arrays p (T, N, 2), theta (T, N), speed (T, N), and cont (T,): sample t
    directly follows sample t - 1 in time. stride counts 10 Hz rows."""
    cache = DATA.parent.parent / f"{flight}_s{stride}.npz"
    if cache.exists():
        return dict(np.load(cache))
    arrs = [np.loadtxt(f, comments="#", usecols=range(11))
            for f in sorted((DATA / flight).iterdir())]
    t0 = max(a[0, 0] for a in arrs)
    t1 = min(a[-1, 0] for a in arrs)
    cols = [a[(a[:, 0] >= t0) & (a[:, 0] <= t1)][::stride] for a in arrs]
    n = min(len(c) for c in cols)
    a = np.stack([c[:n] for c in cols], 1)
    speed = np.hypot(a[..., 4], a[..., 5])
    keep = (speed > MIN_FLY_SPEED).all(1)
    idx = np.flatnonzero(keep)
    cont = np.zeros(len(idx), bool)
    cont[1:] = np.diff(idx) == 1
    out = {"p": a[idx][..., 1:3], "theta": np.arctan2(a[idx][..., 5], a[idx][..., 4]),
           "speed": speed[idx], "cont": cont}
    np.savez_compressed(cache, **out)
    return out
