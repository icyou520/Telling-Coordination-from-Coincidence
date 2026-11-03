"""Experiment 12: confirmatory test of local order as the missing macro variable.

Experiment 11 falsified band strength as the variable that closes the flock's
layer, but found post hoc that the local-alignment histogram does much better
(autonomy of d(phi) 0.68 -> 0.85 below the transition). Because that was
selected from 20 candidates on the same data, it is tested here on fresh seeds
with a single variable fixed in advance:

    L = mean local order = mean over agents of |mean heading within radius 1|
        (from the 8-bin local-order histogram, bin centres).

Physical reading: L - phi measures how much the flock is aligned locally but
not globally (domains moving in different directions); domains merging is
what changes phi next.

Predictions, fixed before running (fresh seeds 100..105, N = 1600):

  L1  Mean autonomy of d(phi) with (phi, L) at noises 2.0 .. chi peak >= 0.8.
  L2  Autonomy of d(phi) with (phi, L) at the chi peak >= 0.8.
  L3  The new layer is closed: autonomy of d(L) with (phi, L) at the chi peak
      >= 0.8 (undefined counts as failure).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e11_band_closure import N, NOISES, _autonomy, _mean, _simulate

SEEDS = range(100, 106)
CENTRES = (np.arange(8) + 0.5) / 8


def score(noise, seed, phi, feats) -> dict:
    local = feats[:, :8] @ CENTRES
    c2 = np.column_stack([phi, local])
    return {"noise": float(noise), "seed": int(seed), "chi": float(N * phi.var()),
            "local_mean": float(local.mean()), "pol_mean": float(phi.mean()),
            "a_phi_c1": _autonomy(phi[:, None], np.diff(phi), feats),
            "a_phi_c2": _autonomy(c2, np.diff(phi), feats),
            "a_local_c2": _autonomy(c2, np.diff(local), feats)}


def _score_args(args):
    return score(*args)


def analyze(rows: list[dict], out_dir: Path) -> str:
    def curve(key):
        return np.array([_mean([r[key] for r in rows if r["noise"] == e]) for e in NOISES])

    chi = curve("chi")
    ic = int(np.nanargmax(chi))
    a1, a2, al = curve("a_phi_c1"), curve("a_phi_c2"), curve("a_local_c2")
    pol, loc = curve("pol_mean"), curve("local_mean")
    lines = [f"Experiment 12: local order as a second macro variable (N={N}, fresh seeds)", "",
             f"  chi peak at noise {NOISES[ic]}", "",
             f"  {'noise':>6}{'phi':>7}{'L':>7}{'A dphi | phi':>14}{'A dphi | phi,L':>16}"
             f"{'A dL | phi,L':>14}"]
    for i, e in enumerate(NOISES):
        mark = " <- chi peak" if i == ic else ""
        lines.append(f"  {e:6.2f}{pol[i]:7.3f}{loc[i]:7.3f}{a1[i]:14.3f}{a2[i]:16.3f}"
                     f"{al[i]:14.3f}{mark}")
    window = slice(0, ic + 1)
    m1, m2 = _mean(a1[window]), _mean(a2[window])
    l1 = np.isfinite(m2) and m2 >= 0.8
    l2 = np.isfinite(a2[ic]) and a2[ic] >= 0.8
    l3 = np.isfinite(al[ic]) and al[ic] >= 0.8
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(l1)}  L1 mean autonomy of d(phi) with (phi, L) below the peak >= 0.8 "
              f"[{m2:.3f}; phi alone {m1:.3f}]",
              f"  {tag(l2)}  L2 autonomy of d(phi) with (phi, L) at the peak >= 0.8 [{a2[ic]:.3f}]",
              f"  {tag(l3)}  L3 local order itself closed at the peak >= 0.8 [{al[ic]:.3f}]"]
    _plot(a1, a2, al, ic, out_dir / "e12_local_order_closure.png")
    lines.append(f"\n  figure: {out_dir / 'e12_local_order_closure.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(a1, a2, al, ic, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(NOISES, a1, "o-", label="d(phi) | phi")
    ax.plot(NOISES, a2, "s-", label="d(phi) | phi, L")
    ax.plot(NOISES, al, "^--", label="d(L) | phi, L")
    ax.axvline(NOISES[ic], color="tab:red", ls=":", lw=1)
    ax.set_xlabel("noise")
    ax.set_ylabel("autonomy")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f"Closure with local order as second macro variable, N={N}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(SEEDS)[:1] if quick else list(SEEDS)
    tasks = [(N, e, s, "flock") for e in NOISES for s in seeds]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"  running {len(tasks)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        sims = list(pool.map(_simulate, tasks, chunksize=1))
        np.savez_compressed(out_dir / "e12_trajectories.npz",
                            noise=np.array([s[0] for s in sims]), seed=np.array([s[1] for s in sims]),
                            phi=np.array([s[2] for s in sims]), feats=np.array([s[3] for s in sims]))
        rows = list(pool.map(_score_args, sims, chunksize=1))
    (out_dir / "e12_local_order_closure_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows, out_dir)


if __name__ == "__main__":
    main()
