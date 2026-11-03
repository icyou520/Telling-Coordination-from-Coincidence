"""Loader for the ALM recordings with midbrain (PPN/MRN) perturbation of Inagaki, Chen et al. (2022, Cell 185,
1065-1081; DANDI 000221, CC-BY 4.0).

Files go in data/inagaki/<Session_name>.nwb with the authors' Session_spreadsheet.csv. Spike times are relative to
each trial's start. PhotostimulationType 0 = control, 3 = PPN/MRN perturbation (0.25 mW), light from 0.6 s before
to 0.6 s after the go cue (the authors' plotNWBdata_PPNsilencing.m).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "inagaki"
DT = 0.01
PRE, POST = 2.0, 1.0
CUE_BIN = int(round(PRE / DT))
NB = int(round((PRE + POST) / DT))
CONTROL, SILENCED = 0, 3


def sessions() -> list[dict]:
    """ALM sessions with PPN/MRN perturbation, from the authors' spreadsheet."""
    rows = csv.DictReader(open(DATA / "Session_spreadsheet.csv", encoding="utf-8-sig"))
    return [r for r in rows if r["Exp_type"].startswith("PPN/MRN perturbation") and r["Recording_site"] == "ALM"]


def load(name: str) -> dict:
    """Cue-aligned spike counts (units, trials, bins) for the units recorded on every kept trial."""
    import h5py
    f = h5py.File(DATA / f"{name}.nwb", "r")
    t = f["intervals/trials"]
    ids = t["id"][:]
    ptype = t["PhotostimulationType"][:]
    keep = (t["GoodTrials"][:] > 0) & np.isin(ptype, [CONTROL, SILENCED])
    idx = f["units/spike_times_index"][:]
    times, tid = f["units/spike_times"][:], f["units/trialsID"][:]
    bounds = np.split(np.arange(len(times)), idx[:-1])
    lo = np.array([tid[b].min() if len(b) else np.inf for b in bounds])
    hi = np.array([tid[b].max() if len(b) else -np.inf for b in bounds])
    cover = np.array([((lo <= i) & (hi >= i)).sum() for i in ids])
    window = ids[cover >= 0.9 * cover.max()]
    units = np.flatnonzero((lo <= window.min()) & (hi >= window.max()))
    trials = ids[keep & np.isin(ids, window)]
    row = {i: k for k, i in enumerate(trials)}
    cue = dict(zip(ids, t["CueTime"][:]))
    x = np.zeros((len(units), len(trials), NB), dtype=np.int16)
    for j, u in enumerate(units):
        b = bounds[u]
        for s, tr in zip(times[b], tid[b]):
            k = row.get(tr)
            if k is None:
                continue
            bin_ = int(np.floor((s - cue[tr] + PRE) / DT))
            if 0 <= bin_ < NB:
                x[j, k, bin_] += 1
    pos = {i: k for k, i in enumerate(ids)}
    sel = np.array([pos[i] for i in trials])
    left = (t["HitL"][:] | t["ErrL"][:] | t["NoLickL"][:])[sel].astype(bool)
    return {"x": x, "silenced": ptype[sel] == SILENCED, "left": left, "trials": trials}
