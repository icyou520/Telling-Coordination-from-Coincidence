"""Experiment 92: Onsager vortex clusters in a two-dimensional superfluid (point-vortex simulation).

In a flat superfluid (a thin Bose-Einstein condensate, or a helium film) the flow is carried by quantized vortices,
each with circulation +1 or -1. Their motion follows the point-vortex equations, the long-distance limit of the
Gross-Pitaevskii equation: every vortex is carried by the flow of all the others and of its own and their images in
the wall. Onsager (1949) predicted that at high energy, same-sign vortices gather into large clusters (negative
temperature states); at low energy they bind into +/- pairs. Gauthier et al. and Johnstone et al. (Science 364,
2019) observed such clusters in disk- and stadium-shaped condensates.
The question: told only each vortex's velocity, does the share see the cluster state as collective motion, and stay
near zero for vortices that do not interact? If so, it is a model-free order parameter for vortex tracking data.

Model: 20 vortices (10 of each sign; 8 in the small-group case) in the unit disk, wall images included, cores
softened at 0.03, 4th-order Runge-Kutta with dt 0.005, weak positional diffusion (0.002 per unit time^1/2) to stand
for thermal noise; 2,000 frames, one every 0.05 time units (500 frames for small groups). Preparations, from low to
high energy per vortex: dipoles (pairs 0.06 apart), random (uniform in the disk), and clusters (one Gaussian blob of
each sign, centred at +/-0.45, spread 0.35, 0.25, 0.15 or 0.08). The non-interacting control keeps only each
vortex's own image (it precesses on its own). Members: each vortex's frame-to-frame x velocity, and separately its
y velocity; S is the mean of the two. No recorded causes. Design: horizon 1 frame, windows (1, 5, 22). Virtual
groups: each member circularly shifted in time (twinshare's shift). Imaging jitter: Gaussian error of 0.01 (a third
of a core) added to every position before differencing.

Development (seeds 0 and 10-13; no prediction seed was run):
  * positions as members read S +0.12 for non-interacting vortices (smooth precession), so velocities are used;
  * non-interacting: |S| <= 0.014, with or without jitter; virtual groups <= 0.009 (20 vortices);
  * dipoles +0.03..+0.43 and random +0.07..+0.64: interacting, but the share varies widely between runs;
  * clusters: spread 0.35 +0.73, 0.25 +0.85..+0.95, 0.15 +0.97..+0.99, 0.08 +0.999;
  * jitter 0.01: clusters 0.15 +0.85..+0.95, clusters 0.25 +0.44..+0.68;
  * 8 vortices, 500 frames: clusters 0.15 +0.98..+1.00 against random +0.03..+0.19; virtual <= 0.032; with jitter
    clusters fell to +0.39..+0.85.

Pre-registered predictions (seeds 1-3, run once):
  P1  Non-interacting vortices: |S| <= 0.03, with and without jitter (each seed).
  P2  Virtual groups: S <= 0.02 for 20 vortices, <= 0.05 for 8 (every case and seed).
  P3  Clusters: S > 0.8 at spread 0.25 and S > 0.9 at spreads 0.15 and 0.08 (each seed).
  P4  Every cluster state of spread 0.25 or less has a larger share than every dipole and random state (each seed).
  P5  With imaging jitter 0.01, clusters of spread 0.15 keep S > 0.75 (each seed).
  P6  8 vortices over 500 frames: clusters (spread 0.15) S > 0.9, and at least 0.5 above random (each seed).
Reported, not predicted: S against energy per vortex, the memory gap S_pred, jitter for small groups.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twinshare" / "src"))
from twinshare import engine as E  # noqa: E402

DESIGN = E.Design(windows=(1, 5, 22))
CORE, DT, EVERY, DIFFUSION, WALL = 0.03, 0.005, 0.05, 0.002, 0.97
JITTER = 0.01
SEEDS = (1, 2, 3)
CASES = {                                              # name: (preparation, keyword arguments)
    "non-interacting": ("random", dict(interact=False)),
    "dipoles": ("dipoles", {}),
    "random": ("random", {}),
    "clusters 0.35": ("clusters", dict(spread=0.35)),
    "clusters 0.25": ("clusters", dict(spread=0.25)),
    "clusters 0.15": ("clusters", dict(spread=0.15)),
    "clusters 0.08": ("clusters", dict(spread=0.08)),
    "small random": ("random", dict(n=4, frames=500)),
    "small clusters 0.15": ("clusters", dict(spread=0.15, n=4, frames=500)),
}


# ---------------------------------------------------------------- point vortices in a disk

def velocity(z, k, interact=True):
    """dz/dt of vortices at complex positions z with circulations k in the unit disk (images at 1/conj(z))."""
    img = 1.0 / np.conj(z)
    if interact:
        d_img = z[:, None] - img[None, :]
        w = -(k[None, :] * np.conj(d_img) / (np.abs(d_img) ** 2 + CORE ** 2)).sum(1)
        d = z[:, None] - z[None, :]
        w = w + ((1.0 - np.eye(len(z))) * k[None, :] * np.conj(d) / (np.abs(d) ** 2 + CORE ** 2)).sum(1)
    else:
        w = -k * np.conj(z - img) / (np.abs(z - img) ** 2 + CORE ** 2)
    return np.conj(w / (2j * np.pi))


def energy(z, k):
    """Point-vortex energy per vortex in the unit disk (pair terms plus wall images)."""
    d = np.abs(z[:, None] - z[None, :]) + np.eye(len(z))
    kk = k[:, None] * k[None, :]
    e = -(kk * np.log(d)).sum() + (kk * np.log(np.abs(1 - z[:, None] * np.conj(z[None, :])))).sum()
    return float(e / (4 * np.pi) / len(z))


def prepare(kind, rng, n=10, spread=0.15):
    k = np.r_[np.ones(n), -np.ones(n)]
    if kind == "dipoles":
        c = np.sqrt(rng.uniform(0, 0.6, n)) * np.exp(2j * np.pi * rng.random(n))
        a = 0.06 * np.exp(2j * np.pi * rng.random(n))
        z = np.r_[c + a / 2, c - a / 2]
    elif kind == "random":
        z = np.sqrt(rng.uniform(0, 0.8, 2 * n)) * np.exp(2j * np.pi * rng.random(2 * n))
    else:
        z = np.r_[0.45 + spread * (rng.normal(size=n) + 1j * rng.normal(size=n)),
                  -0.45 + spread * (rng.normal(size=n) + 1j * rng.normal(size=n))]
    return np.where(np.abs(z) > 0.9, 0.9 * z / np.abs(z), z), k


def simulate(kind, rng, frames=2000, interact=True, **kw):
    """Trajectories (frames, vortices) as complex positions, circulations, and the initial energy per vortex."""
    z, k = prepare(kind, rng, **kw)
    e0 = energy(z, k)
    out = np.empty((frames, len(z)), complex)
    sub = int(round(EVERY / DT))
    for f in range(frames):
        for _ in range(sub):
            a = velocity(z, k, interact)
            b = velocity(z + DT / 2 * a, k, interact)
            c = velocity(z + DT / 2 * b, k, interact)
            d = velocity(z + DT * c, k, interact)
            z = z + DT / 6 * (a + 2 * b + 2 * c + d)
            z = z + DIFFUSION * np.sqrt(DT) * (rng.normal(size=len(z)) + 1j * rng.normal(size=len(z)))
            r = np.abs(z)
            z = np.where(r > WALL, WALL * z / r, z)
        out[f] = z
    return out, k, e0


# ---------------------------------------------------------------- measures

def measure(traj, rng, n_virtual=5):
    def shares(p):
        v = np.diff(p, axis=0)
        z0 = np.zeros((len(v), 0))
        return v, z0, float(np.mean([E.share(v.real, z0, DESIGN).S, E.share(v.imag, z0, DESIGN).S]))
    v, z0, s = shares(traj)
    virt = [float(np.mean([E.share(E.shift(v.real, rng), z0, DESIGN).S, E.share(E.shift(v.imag, rng), z0, DESIGN).S]))
            for _ in range(n_virtual)]
    noisy = traj + JITTER * (rng.normal(size=traj.shape) + 1j * rng.normal(size=traj.shape))
    s_pred = float(np.mean([E.memory_gap(v.real, z0, DESIGN)["S_pred"], E.memory_gap(v.imag, z0, DESIGN)["S_pred"]]))
    return {"S": s, "S_jitter": shares(noisy)[2], "S_pred": s_pred, "virtual": virt,
            "vortices": traj.shape[1], "frames": traj.shape[0]}


# ---------------------------------------------------------------- predictions and report

def _get(res, case):
    return [res[f"{case}, seed {s}"] for s in sorted({int(k.rsplit(" ", 1)[1]) for k in res})]


def predictions(res):
    out = []
    ni = _get(res, "non-interacting")
    out.append(("P1", "non-interacting: |S| <= 0.03 with and without jitter",
                all(abs(r["S"]) <= 0.03 and abs(r["S_jitter"]) <= 0.03 for r in ni),
                ", ".join(f"{r['S']:+.3f}/{r['S_jitter']:+.3f}" for r in ni)))
    big = [max(r["virtual"]) for k, r in res.items() if not k.startswith("small")]
    small = [max(r["virtual"]) for k, r in res.items() if k.startswith("small")]
    out.append(("P2", "virtual groups: S <= 0.02 (20 vortices), <= 0.05 (8)",
                max(big) <= 0.02 and max(small) <= 0.05, f"max {max(big):+.3f} / {max(small):+.3f}"))
    c25, c15, c08 = _get(res, "clusters 0.25"), _get(res, "clusters 0.15"), _get(res, "clusters 0.08")
    out.append(("P3", "clusters: S > 0.8 (spread 0.25), > 0.9 (0.15, 0.08)",
                all(r["S"] > 0.8 for r in c25) and all(r["S"] > 0.9 for r in c15 + c08),
                f"0.25 min {min(r['S'] for r in c25):+.3f}; 0.15 min {min(r['S'] for r in c15):+.3f}; "
                f"0.08 min {min(r['S'] for r in c08):+.3f}"))
    ok, detail = True, []
    for i, seed_rows in enumerate(zip(c25, c15, c08, _get(res, "dipoles"), _get(res, "random"))):
        lo_cluster = min(r["S"] for r in seed_rows[:3])
        hi_other = max(r["S"] for r in seed_rows[3:])
        ok &= lo_cluster > hi_other
        detail.append(f"{lo_cluster:+.3f} > {hi_other:+.3f}")
    out.append(("P4", "every cluster (spread <= 0.25) above every dipole and random state", ok, "; ".join(detail)))
    out.append(("P5", "jitter 0.01: clusters 0.15 S > 0.75", all(r["S_jitter"] > 0.75 for r in c15),
                ", ".join(f"{r['S_jitter']:+.3f}" for r in c15)))
    sc, sr = _get(res, "small clusters 0.15"), _get(res, "small random")
    out.append(("P6", "8 vortices, 500 frames: clusters S > 0.9 and >= random + 0.5",
                all(c["S"] > 0.9 and c["S"] - r["S"] >= 0.5 for c, r in zip(sc, sr)),
                ", ".join(f"{c['S']:+.3f} vs {r['S']:+.3f}" for c, r in zip(sc, sr))))
    return out


def analyze(res):
    lines = ["Experiment 92: Onsager vortex clusters in a two-dimensional superfluid (point vortices).", ""]
    for k, r in res.items():
        lines.append(f"  {k:32s} E/N {r['energy']:+.3f}  S {r['S']:+.4f}  jitter {r['S_jitter']:+.4f}  "
                     f"S_pred {r['S_pred']:+.4f}  virtual {min(r['virtual']):+.4f}..{max(r['virtual']):+.4f}")
    e = np.array([r["energy"] for k, r in res.items() if k.split(",")[0] not in ("non-interacting",)
                  and not k.startswith("small")])
    s = np.array([r["S"] for k, r in res.items() if k.split(",")[0] not in ("non-interacting",)
                  and not k.startswith("small")])
    rank = np.corrcoef(np.argsort(np.argsort(e)), np.argsort(np.argsort(s)))[0, 1]
    lines += ["", f"  Exploratory: rank correlation of S with energy per vortex (20 interacting vortices): {rank:+.3f}",
              "", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    """quick: one fresh seed (a structural smoke test)."""
    out_dir = out_dir or Path("results")
    res = {}
    for s in ((101,) if quick else SEEDS):
        for i, (name, (kind, kw)) in enumerate(CASES.items()):
            rng = np.random.default_rng([92, s, i])
            traj, k, e0 = simulate(kind, rng, **kw)
            r = measure(traj, rng)
            r["energy"] = e0
            res[f"{name}, seed {s}"] = r
    (out_dir / "e92_vortices_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e92_figures import make_figures
    make_figures(res, out_dir / "e92_figures", text)
    return text


if __name__ == "__main__":
    main()
