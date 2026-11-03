"""Loader for the Neuropixels recordings of Steinmetz et al. (2019, Nature 576,
266-273), in the trial-binned form distributed by Neuromatch Academy
(https://osf.io/agvxh, uv3mw, ehmw2; CC-BY 4.0).

Files go in data/steinmetz/steinmetz_part{0,1,2}.npz. Each of the 39 sessions
holds spike counts in 10 ms bins, 250 bins per trial, stimulus onset at bin 50.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "steinmetz"
DT = 0.01
STIM_BIN = 50


N_SESSIONS = 39


def _split() -> None:
    """Write one cache file per session, so workers need not load all three parts."""
    s = 0
    for j in range(3):
        for d in np.load(DATA / f"steinmetz_part{j}.npz", allow_pickle=True)["dat"]:
            np.save(DATA / f"session_{s:02d}.npy", np.array([d], dtype=object))
            s += 1


@lru_cache(maxsize=2)
def session(s: int) -> dict:
    path = DATA / f"session_{s:02d}.npy"
    if not path.exists():
        _split()
    return np.load(path, allow_pickle=True)[0]


def areas(min_neurons: int = 30) -> list[tuple[int, str, int]]:
    """(session, area, neuron count) for every area with enough neurons."""
    out = []
    for s in range(N_SESSIONS):
        names, counts = np.unique(session(s)["brain_area"], return_counts=True)
        for a, c in zip(names, counts):
            if a != "root" and c >= min_neurons:
                out.append((s, str(a), int(c)))
    return out


def trial_data(s: int) -> dict:
    """Per-trial covariates of session s, as float arrays."""
    d = session(s)
    face = d["face"][0] if d["face"].ndim == 3 else d["face"]
    return {
        "cl": d["contrast_left"].astype(float),
        "cr": d["contrast_right"].astype(float),
        "gocue": d["gocue"].ravel().astype(float),
        "feedback": d["feedback_time"].ravel().astype(float),
        "wheel": d["wheel"][0].astype(float),
        "pupil": d["pupil"][0].astype(float),
        "face": face.astype(float),
        "licks": d["licks"][0].astype(float),
    }


def spikes(s: int, area: str | None = None) -> np.ndarray:
    """Spike counts (neurons, trials, bins) of session s, optionally one area."""
    d = session(s)
    if area is None:
        return d["spks"]
    return d["spks"][d["brain_area"] == area]
