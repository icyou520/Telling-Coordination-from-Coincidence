"""Lafoux et al. (2023): schools of about 50 rummy-nose tetras at fixed light levels (figshare 22434919).

    curl -L -o data/lafoux/discrete_light_variation.zip https://ndownloader.figshare.com/files/39883996
    unzip data/lafoux/discrete_light_variation.zip -d data/lafoux/discrete

The fixed-light control set: one hour per recording (5 min habituation, 50 min recorded at 10 frames per second),
normalized illuminance E in {0, 0.05, 0.1, 0.15, 0.3, 0.5, 0.75, 1} (1 = 900 lux), two replicates with FastTrack
trajectories each (a third replicate at 5 fps has no trajectories). Tracks: FastTrack ids; an id is lost and a new one
assigned when a fish is not followed (250-1,100 ids per recording), and an id may miss frames.
Rule fixed before any share was computed: a recording is used only if its metadata light level matches its name
(Tracking_Result_E1-0_fps10_T3000_0 says 0.3 and is excluded).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path("data/lafoux/discrete/discrete_data/supplementary_exp_fixed_light/tracking_results")
CACHE = Path("data/lafoux/cache")
DT = 0.1


def _meta(d):
    out = {}
    for line in (d / "metadata.yaml").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = float(v)
    return out


def recordings():
    """[(name, light from the name, metadata, used)], sorted by light then replicate."""
    out = []
    for d in sorted(ROOT.glob("Tracking_Result_*")):
        light = float(d.name.split("_")[2][1:].replace("-", "."))
        meta = _meta(d)
        out.append((d.name, light, meta, abs(meta["light_intensity"] - light) < 1e-9))
    return sorted(out, key=lambda r: (r[1], r[2]["replicate"]))


def _raw(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if not f.exists():
        a = np.loadtxt(ROOT / name / "tracking.txt", skiprows=1, usecols=(6, 7, 21, 22))
        np.savez_compressed(f, x=a[:, 0], y=a[:, 1], frame=a[:, 2].astype(np.int32), id=a[:, 3].astype(np.int32))
    z = np.load(f)
    return z["frame"], z["id"], np.column_stack([z["x"], z["y"]])


@lru_cache(maxsize=2)
def tracks(name):
    """(frames, [(start, velocity (L, 2) per second with NaN where a frame or the next one is missing)])."""
    frame, ids, pos = _raw(name)
    t = int(frame.max()) + 1
    out = []
    order = np.lexsort((frame, ids))
    frame, ids, pos = frame[order], ids[order], pos[order]
    cut = np.flatnonzero(np.diff(ids)) + 1
    for fr, p in zip(np.split(frame, cut), np.split(pos, cut)):
        fr, keep = np.unique(fr, return_index=True)
        p = p[keep]
        if fr[-1] - fr[0] < 2:
            continue
        span = np.full((fr[-1] - fr[0] + 1, 2), np.nan)
        span[fr - fr[0]] = p
        out.append((int(fr[0]), np.diff(span, axis=0) / DT))
    return t - 1, out
