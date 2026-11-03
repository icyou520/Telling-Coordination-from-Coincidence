"""Fish data with interaction removed or made one-way, resampled to 10 Hz.

Liu & Burbano (2025), Zenodo 10.5281/zenodo.15351451: rummy-nose tetras (Hemigrammus rhodostomus) alone, in pairs
and in triads, in bright and dark arenas (~10.5 min per recording, 33.3 Hz, pixel coordinates).
Escobedo et al. (2025), figshare 30539666: the same species in a hemispherical bowl. 2R: two real fish. C1, C2: one
real fish with a virtual conspecific that responds to it (C2 is a faster clone). C3: the C2 clone, independent of the
real fish (one-way interaction). 30 Hz, metres, continuous intervals separated by jumps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path("data/intervention")
LIU = ROOT / "liu" / "swarmintelligencelab-IntermittentExploration-2eb7fdd"
VR = ROOT / "vr"
DT = 0.1
SIZES = {"Singles": 1, "Pairs": 2, "Triads": 3}
LIGHTS = ("Bright", "Dark")
VR_CONDITIONS = {"2R": "2R-TwoRealFish.dat", "C1": "C1-OneReal+OneVirtual.dat", "C2": "C2-OneReal+OneVirtual.dat",
                 "C3": "C3-OneReal+OneVirtual.dat"}
MIN_SEGMENT = 30.0


def liu(size: str, light: str) -> list[tuple[str, np.ndarray]]:
    """Recordings of one condition: (name, positions (T, N, 2)) on a 10 Hz grid. A few frames have no position
    (at most 12 per fish, in runs of at most 5 frames); interpolation bridges them."""
    out = []
    for f in sorted((LIU / f"{size}-{light}").glob("*.csv")):
        a = np.genfromtxt(f, delimiter=";", skip_header=1)
        t, xy = a[:, 1], a[:, 2:]
        grid = np.arange(t[0], t[-1], DT)
        ok = ~np.isnan(xy)
        pos = np.stack([np.interp(grid, t[ok[:, c]], xy[ok[:, c], c]) for c in range(xy.shape[1])], 1)
        out.append((f.stem.split("-")[1][:14], pos.reshape(len(grid), -1, 2)))
    return out


def vr(condition: str) -> dict[int, list[np.ndarray]]:
    """Experiments of one VR condition: {experiment: [positions (T, 2, 2), ...]} for continuous intervals of at
    least MIN_SEGMENT s, horizontal plane, every third frame (10 Hz). Member 0 is real; member 1 is virtual (C1-C3)."""
    a = np.loadtxt(VR / VR_CONDITIONS[condition])
    out: dict[int, list[np.ndarray]] = {}
    for e in np.unique(a[:, 0]):
        ae = a[a[:, 0] == e]
        for iv in np.unique(ae[:, 1]):
            s = ae[ae[:, 1] == iv]
            if s[-1, 2] - s[0, 2] < MIN_SEGMENT:
                continue
            s = s[::3]
            out.setdefault(int(e), []).append(np.stack([s[:, 3:5], s[:, 6:8]], 1))
    return out


def velocity(pos: np.ndarray) -> np.ndarray:
    """Planar velocity (T-1, N, 2) per second."""
    return np.diff(pos, axis=0) / DT
