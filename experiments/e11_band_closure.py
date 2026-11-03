"""Experiment 11: does adding band strength restore closure of the flock's layer?

Experiment 10 found that polarization phi alone is not closed near the
transition in a large flock (N = 1600): micro detail predicts its future,
and the leak moves with the transition. Large Vicsek flocks form travelling
density bands there, so the candidate missing macro variable is

    B = band strength = sqrt(sum of squared k=1,2 Fourier amplitudes of the
        agent density along x and along y)   (micro features 16..19).

Macro descriptions compared, each against the same full set of 20 micro
features (local-order histogram, crowding histogram, band amplitudes):

    C1 = phi          (experiment 10)
    C2 = (phi, B)

Autonomy is computed for the change of each macro variable, with a
degree-2 polynomial macro law.

Predictions, fixed before running (chi peak = noise of maximum mean N*Var(phi)):

  H1  Autonomy of d(phi) with C2 at the chi peak >= 0.9.
  H2  Averaged over noises from 2.0 up to the chi peak, autonomy of d(phi)
      rises by >= 0.15 from C1 to C2.
  H3  The new layer is closed too: autonomy of d(B) with C2 at the chi peak
      >= 0.8 (undefined counts as failure).
  H4  Specificity: of the 20 micro features, each tried alone as the second
      macro variable, the one giving the highest mean autonomy of d(phi) at
      noises 2.0 .. chi peak is a band amplitude (features 16..19).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e9_interaction_closure import BLOCK, _run

N = 1600
NOISES = np.round(np.arange(2.0, 5.01, 0.25), 3)
SEEDS = 6
BAND = slice(16, 20)
FEATURE_NAMES = ([f"order[{i}]" for i in range(8)] + [f"crowd[{i}]" for i in range(8)]
                 + ["band x k1", "band x k2", "band y k1", "band y k2"])


def _simulate(args):
    run = _run(args)
    return run["noise"], run["seed"], np.array(run["phi"]), np.array(run["feats"])


def _autonomy(macro: np.ndarray, target: np.ndarray, feats: np.ndarray) -> float:
    from emergence_lab.closure import predictive_closure

    blocks = np.arange(len(target)) // BLOCK
    return predictive_closure(macro[:-1], target, feats[:-1], groups=blocks, degree=2,
                              n_micro_components=feats.shape[1],
                              predict_change=False)["autonomy"]


def score(noise, seed, phi, feats) -> dict:
    band = np.sqrt((feats[:, BAND] ** 2).sum(1))
    dphi, dband = np.diff(phi), np.diff(band)
    c1, c2 = phi[:, None], np.column_stack([phi, band])
    return {
        "noise": float(noise), "seed": int(seed), "chi": float(N * phi.var()),
        "a_phi_c1": _autonomy(c1, dphi, feats),
        "a_phi_c2": _autonomy(c2, dphi, feats),
        "a_band_c2": _autonomy(c2, dband, feats),
        "a_phi_single": [_autonomy(np.column_stack([phi, feats[:, j]]), dphi, feats)
                         for j in range(feats.shape[1])],
    }


def _score_args(args):
    return score(*args)


def _mean(a) -> float:
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def analyze(rows: list[dict], out_dir: Path) -> str:
    def curve(key):
        return np.array([_mean([r[key] for r in rows if r["noise"] == e]) for e in NOISES])

    chi = curve("chi")
    ic = int(np.nanargmax(chi))
    a1, a2, ab = curve("a_phi_c1"), curve("a_phi_c2"), curve("a_band_c2")
    single = np.array([[_mean([r["a_phi_single"][j] for r in rows if r["noise"] == e])
                        for j in range(len(FEATURE_NAMES))] for e in NOISES])
    lines = [f"Experiment 11: band strength as a second macro variable (N={N})", "",
             f"  chi peak at noise {NOISES[ic]}", "",
             f"  {'noise':>6}{'chi':>8}{'A dphi | phi':>14}{'A dphi | phi,B':>16}{'A dB | phi,B':>14}"]
    for i, e in enumerate(NOISES):
        mark = " <- chi peak" if i == ic else ""
        lines.append(f"  {e:6.2f}{chi[i]:8.2f}{a1[i]:14.3f}{a2[i]:16.3f}{ab[i]:14.3f}{mark}")

    window = slice(0, ic + 1)
    gain = _mean(a2[window]) - _mean(a1[window])
    ranking = np.argsort([-_mean(single[window, j]) for j in range(len(FEATURE_NAMES))])
    lines += ["", "  second macro variable, mean autonomy of d(phi) at noises 2.0 .. chi peak:"]
    lines += [f"    {FEATURE_NAMES[j]:<10} {_mean(single[window, j]):.3f}" for j in ranking[:6]]
    lines.append(f"    {'B (bands)':<10} {_mean(a2[window]):.3f}")
    lines.append(f"    {'none':<10} {_mean(a1[window]):.3f}")

    h1 = np.isfinite(a2[ic]) and a2[ic] >= 0.9
    h2 = np.isfinite(gain) and gain >= 0.15
    h3 = np.isfinite(ab[ic]) and ab[ic] >= 0.8
    h4 = 16 <= ranking[0] <= 19
    tag = lambda ok: "PASS" if ok else "FAIL"
    lines += ["", "  Pre-registered predictions:",
              f"  {tag(h1)}  H1 autonomy of d(phi) with (phi, B) at the peak >= 0.9 [{a2[ic]:.3f}]",
              f"  {tag(h2)}  H2 mean gain from adding B >= 0.15 below the peak [{gain:+.3f}]",
              f"  {tag(h3)}  H3 band strength itself closed at the peak >= 0.8 [{ab[ic]:.3f}]",
              f"  {tag(h4)}  H4 best single second variable is a band amplitude "
              f"[{FEATURE_NAMES[ranking[0]]}]"]
    _plot(a1, a2, ab, single, ranking, ic, out_dir / "e11_band_closure.png")
    lines.append(f"\n  figure: {out_dir / 'e11_band_closure.png'}")
    text = "\n".join(lines)
    print(text)
    return text


def _plot(a1, a2, ab, single, ranking, ic, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(NOISES, a1, "o-", label="d(phi) | phi")
    ax[0].plot(NOISES, a2, "s-", label="d(phi) | phi, B")
    ax[0].plot(NOISES, ab, "^--", label="d(B) | phi, B")
    ax[0].axvline(NOISES[ic], color="tab:red", ls=":", lw=1)
    ax[0].set_xlabel("noise")
    ax[0].set_ylabel("autonomy")
    ax[0].set_ylim(0.3, 1.05)
    ax[0].set_title(f"Closure of the flock layer, N={N}")
    ax[0].legend(fontsize=8)
    window = slice(0, ic + 1)
    vals = [_mean(single[window, j]) for j in ranking]
    colors = ["tab:orange" if 16 <= j <= 19 else "tab:gray" for j in ranking]
    ax[1].barh([FEATURE_NAMES[j] for j in ranking][::-1], vals[::-1], color=colors[::-1])
    ax[1].axvline(_mean(a1[window]), color="k", ls=":", lw=1, label="phi alone")
    ax[1].set_xlim(0.3, 1.0)
    ax[1].set_title("Second macro variable: autonomy of d(phi), noise <= peak")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    import json

    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = 1 if quick else SEEDS
    tasks = [(N, e, s, "flock") for e in NOISES for s in range(seeds)]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    print(f"  running {len(tasks)} simulations ...", flush=True)
    workers = max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        sims = list(pool.map(_simulate, tasks, chunksize=1))
        np.savez_compressed(out_dir / "e11_trajectories.npz",
                            noise=np.array([s[0] for s in sims]), seed=np.array([s[1] for s in sims]),
                            phi=np.array([s[2] for s in sims]), feats=np.array([s[3] for s in sims]))
        rows = list(pool.map(_score_args, sims, chunksize=1))
    (out_dir / "e11_band_closure_raw.json").write_text(json.dumps(rows, indent=1))
    return analyze(rows, out_dir)


if __name__ == "__main__":
    main()
