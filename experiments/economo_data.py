"""Loader for the ALM recordings of Economo, Viswanathan, et al. (2018, Nature 563, 79-84; DANDI 000006, CC-BY 4.0).

Files go in data/dandi000006/*.nwb. Pole-location discrimination with a 1.5 s delay and an auditory go cue; no
photostimulation in the released sessions. Trial times (pole, cue) are relative to each trial's start_time;
spike times are absolute. Counts use the same layout as inagaki_data: 10 ms bins from 2 s before to 1 s after the
go cue, cue at bin 200.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "dandi000006"
DT = 0.01
PRE, POST = 2.0, 1.0
CUE_BIN = int(round(PRE / DT))
NB = int(round((PRE + POST) / DT))


def sessions() -> list[str]:
    return sorted(p.stem for p in DATA.glob("*.nwb"))


def load(name: str) -> dict:
    """Counts (units, trials, bins) for good, non-early-lick trials, and the units that fire across them
    (first spike before the first 10% of kept trials, last spike after the last 10%)."""
    import h5py
    f = h5py.File(DATA / f"{name}.nwb", "r")
    t = f["intervals/trials"]
    start, cue = t["start_time"][:], t["cue_start_time"][:]
    resp = t["response"][:].astype(str)
    keep = (t["is_good"][:] > 0) & (resp != "early lick") & np.isfinite(cue)
    cue_abs = (start + cue)[keep]
    left = (t["type"][:].astype(str) == "lick left")[keep]
    idx = f["units/spike_times_index"][:]
    times = f["units/spike_times"][:]
    bounds = np.split(times, idx[:-1])
    lo, hi = np.quantile(cue_abs, [0.1, 0.9])
    units = [s for s in bounds if len(s) and s.min() <= lo and s.max() >= hi]
    x = np.zeros((len(units), len(cue_abs), NB), dtype=np.int16)
    edges = np.arange(NB + 1) * DT - PRE
    for j, s in enumerate(units):
        for k, c in enumerate(cue_abs):
            sel = s[(s >= c - PRE) & (s < c + POST)] - c
            x[j, k] = np.histogram(sel, edges)[0]
    return {"x": x, "left": left, "silenced": np.zeros(len(cue_abs), bool)}
