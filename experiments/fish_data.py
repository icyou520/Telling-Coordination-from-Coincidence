"""Loader for the schooling-fish trajectories of Puy et al. (2024, PNAS 121,
e2309733121; data: https://doi.org/10.5281/zenodo.10890112, CC-BY 4.0).

Files go in data/puy2024/ (see README). Positions are converted to cm
(2745 px = 100 cm); frames are at 50 per second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "puy2024"
FPS = 50
PX_PER_CM = 27.45
RECORDINGS = {39: (0, 1), 8: (0, 1, 2, 3, 4, 5)}


def load(n: int, rec: int) -> np.ndarray:
    """Positions in cm, shape (frames, fish, 2); NaN where untracked."""
    cache = DATA / f"N{n}_rec{rec}.npz"
    if cache.exists():
        return np.load(cache)["pos"]
    raw = np.loadtxt(DATA / f"Experimental_school_N={n}_recording_{rec}.csv",
                     delimiter=",", skiprows=1)
    pos = raw[:, 1:].reshape(len(raw), n, 2) / PX_PER_CM
    np.savez_compressed(cache, pos=pos)
    return pos
