"""Experiment 50: is the school's state closed against its internal configuration?
And do the fish at the front steer the school?

e49 showed that, in real 8-fish schools, the fish's own tendencies add nothing
to forecasting the school's state Z (centroid, mean heading vector, mean
speed, polarization, spread, milling), while in virtual groups they do. It
did not test the school's internal configuration: who is where, heading
which way, at what speed, relative to the school.

School frame: origin at the centroid, x' along the mean heading Theta, y' to
its left. For each fish: (x', y'), relative heading phi = theta - Theta,
relative speed u = v - mean speed. Configuration features (20, symmetric in
the fish, so identity does not matter): means of x'^2, y'^2, x'y', x'^3,
x'y'^2, x' sin phi, y' sin phi, x' cos phi, y' cos phi, sin^2 phi, u x', u y',
u sin phi, u^2, x'^2 sin phi, y'^2 sin phi; nearest-neighbour distance mean
and sd; largest distance to the centroid; front-back extent (max x' - min x').

Part 1, closure (data only, fitted on the first 70%, measured on the last 30%):
  block features = e49's (Z_t ... Z_{t-4}, quadratic terms of Z_t, the school's
  place-and-heading map); block + configuration = the above plus the 20
  configuration features and their squares.
  G(h) = 1 - tr(V^-1 MSE_block+conf) / tr(V^-1 MSE_block), computed separately
  for the motion sub-block (centroid, mean heading vector, mean speed) and the
  shape sub-block (polarization, spread, milling), V = Cov of the change.

Part 2, leadership: L_t = mean(x' sin phi) / spread (fish ahead of the
centroid heading to the left of the mean -> L > 0). Regression of the change
of Theta over 1 s on [1, Theta_t - Theta_{t-1}, L_t], all valid steps,
Newey-West standard errors (30 lags).

Predictions, fixed before running:
  C1  Configuration informs the school's shape: G_shape(1 s) >= 0.05 in >= 5 of
      6 real 8-fish schools.
  C2  Configuration informs the school's motion: G_motion(1 s) >= 0.02 in >= 5
      of 6 real 8-fish schools.
  L1  Front fish steer: the L coefficient is positive with t >= 3 in >= 5 of 6
      real 8-fish schools and in both 39-fish recordings.
  L2  Control: |t| < 3 in >= 4 of 6 virtual 8-fish groups, and the real
      school's standardized L coefficient exceeds its virtual pair's in >= 5
      of 6.
Exploratory: the same closure in virtual groups and in 39 fish; speed-
leadership (mean of u x' > 0: front fish faster, as Puy et al. report).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e48_fish_layers import _lstsq, ratio_gap
from experiments.e49_school_layer import LAGS, Z_NAMES, school_state

SEED = 50
HORIZONS = (10, 30)
NW_LAGS = 30
MOTION, SHAPE = (0, 1, 2, 3, 4), (5, 6, 7)
CONF_NAMES = ("x2", "y2", "xy", "x3", "xy2", "x_sin", "y_sin", "x_cos", "y_cos", "sin2", "u_x", "u_y",
              "u_sin", "u2", "x2_sin", "y2_sin", "nnd_mean", "nnd_sd", "r_max", "extent")


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def configuration(p, th, sp):
    c = p.mean(1)
    big_th = np.arctan2(np.sin(th).mean(1), np.cos(th).mean(1))
    r = p - c[:, None, :]
    ca, sa = np.cos(big_th)[:, None], np.sin(big_th)[:, None]
    x = r[..., 0] * ca + r[..., 1] * sa
    y = -r[..., 0] * sa + r[..., 1] * ca
    phi = _wrap(th - big_th[:, None])
    s, co = np.sin(phi), np.cos(phi)
    u = sp - sp.mean(1, keepdims=True)
    d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)
    d[:, np.arange(p.shape[1]), np.arange(p.shape[1])] = np.inf
    nnd = d.min(-1)
    m = lambda a: a.mean(1)  # noqa: E731
    feats = np.stack([m(x * x), m(y * y), m(x * y), m(x ** 3), m(x * y * y), m(x * s), m(y * s), m(x * co),
                      m(y * co), m(s * s), m(u * x), m(u * y), m(u * s), m(u * u), m(x * x * s), m(y * y * s),
                      nnd.mean(1), nnd.std(1), np.linalg.norm(r, axis=-1).max(1), x.max(1) - x.min(1)], 1)
    spread = np.linalg.norm(r, axis=-1).mean(1)
    lead = m(x * s) / np.maximum(spread, 1e-6)
    return feats, lead, big_th


def school_map(z):
    import experiments.e43_elimination_fish_maps as base
    ang = np.arctan2(z[:, 3], z[:, 2])
    b = base.bumps(z[:, :2])
    return np.concatenate([b, b * np.sin(ang)[:, None], b * np.cos(ang)[:, None]], 1)


def closure(z, conf, train, test, h):
    own = school_map(z)
    iu = np.triu_indices(z.shape[1])

    def block_f(rows):
        cols = [np.ones((len(rows), 1))] + [z[rows - j] for j in range(LAGS)]
        cols += [(z[rows][:, :, None] * z[rows][:, None, :])[:, iu[0], iu[1]], own[rows]]
        return np.concatenate(cols, 1)

    def conf_f(rows):
        return np.concatenate([block_f(rows), conf[rows], conf[rows] ** 2], 1)

    d_tr, d_te = z[train + h] - z[train], z[test + h] - z[test]
    e_b = d_te - _lstsq(block_f(train), d_tr, block_f(test))
    e_c = d_te - _lstsq(conf_f(train), d_tr, conf_f(test))
    out = {}
    for name, idx in (("motion", MOTION), ("shape", SHAPE), ("all", tuple(range(z.shape[1])))):
        i = list(idx)
        v = np.cov(d_te[:, i].T)
        out[name] = ratio_gap(v, e_c[:, i].T @ e_c[:, i] / len(e_c), e_b[:, i].T @ e_b[:, i] / len(e_b))
    out["per_variable"] = (1 - (e_c ** 2).mean(0) / (e_b ** 2).mean(0)).tolist()
    return out


def newey_west(x, y, lags):
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    e = y - x @ beta
    xe = x * e[:, None]
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        g = xe[lag:].T @ xe[:-lag]
        s += (1 - lag / (lags + 1)) * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def leadership(big_th, lead, rows, h=10):
    d_th = _wrap(big_th[rows + h] - big_th[rows])
    rate = _wrap(big_th[rows] - big_th[rows - 1])
    xs = np.column_stack([np.ones(len(rows)), rate, lead[rows]])
    beta, se = newey_west(xs, d_th, NW_LAGS)
    std_coef = beta[2] * lead[rows].std() / d_th.std()
    return {"coef": float(beta[2]), "t": float(beta[2] / se[2]), "std_coef": float(std_coef)}


def _run(args):
    n, rec, kind = args
    from experiments.e31_real_fish import kinematics, virtual
    from experiments.fish_data import load
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    seam = (sp > 100.0).any(1)
    sp = np.where(sp > 100.0, 0.0, sp)
    T = len(th)
    cut = int(0.7 * T)
    hmax = max(HORIZONS)
    bad = np.zeros(T, bool)
    for s in np.nonzero(seam)[0]:
        bad[max(0, s - hmax - 1):s + LAGS + 2] = True
    rows = np.arange(LAGS, T - hmax)
    rows = rows[~bad[rows]]
    train, test = rows[rows + hmax < cut], rows[rows >= cut]
    z = school_state(p, th, sp)
    conf, lead, big_th = configuration(p, th, sp)
    return {"N": n, "rec": rec, "kind": kind,
            "G": {str(h): closure(z, conf, train, test, h) for h in HORIZONS},
            "lead": leadership(big_th, lead, rows),
            "speed_lead": float(conf[rows, CONF_NAMES.index("u_x")].mean()),
            "speed_lead_t": float(conf[rows, CONF_NAMES.index("u_x")].mean()
                                  / newey_west(np.ones((len(rows), 1)), conf[rows, CONF_NAMES.index("u_x")],
                                               NW_LAGS)[1][0])}


def _sel(res, n, kind):
    return sorted([r for r in res["runs"] if r["N"] == n and r["kind"] == kind], key=lambda r: r["rec"])


def predictions(res):
    real, virt, big = _sel(res, 8, "real"), _sel(res, 8, "virtual"), _sel(res, 39, "real")
    c1 = sum(r["G"]["10"]["shape"] >= 0.05 for r in real)
    c2 = sum(r["G"]["10"]["motion"] >= 0.02 for r in real)
    l1a = sum(r["lead"]["coef"] > 0 and r["lead"]["t"] >= 3 for r in real)
    l1b = sum(r["lead"]["coef"] > 0 and r["lead"]["t"] >= 3 for r in big)
    l2a = sum(abs(v["lead"]["t"]) < 3 for v in virt)
    l2b = sum(r["lead"]["std_coef"] > v["lead"]["std_coef"] for r, v in zip(real, virt))
    return [
        ("C1", "configuration informs shape: G_shape(1 s) >= 0.05 (>= 5/6 real)", c1 >= 5,
         "G_shape " + ", ".join(f"{r['G']['10']['shape']:.3f}" for r in real)),
        ("C2", "configuration informs motion: G_motion(1 s) >= 0.02 (>= 5/6 real)", c2 >= 5,
         "G_motion " + ", ".join(f"{r['G']['10']['motion']:.3f}" for r in real)),
        ("L1", "front fish steer: L coefficient > 0, t >= 3 (>= 5/6 real 8; 2/2 39)", l1a >= 5 and l1b == 2,
         "t " + ", ".join(f"{r['lead']['t']:.1f}" for r in real) + "; 39 fish "
         + ", ".join(f"{r['lead']['t']:.1f}" for r in big)),
        ("L2", "control: virtual |t| < 3 (>= 4/6), real std coef > virtual (>= 5/6)", l2a >= 4 and l2b >= 5,
         "virtual t " + ", ".join(f"{v['lead']['t']:.1f}" for v in virt) + f"; real > virtual {l2b}/6"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 50: is the school's state closed against its internal configuration? Do front fish steer?",
             "", "  G_motion / G_shape at 1 s and 3 s; leadership coefficient (t); speed leadership mean(u x')"]
    for r in sorted(res["runs"], key=lambda r: (r["N"], r["kind"], r["rec"])):
        g1, g3 = r["G"]["10"], r["G"]["30"]
        lines.append(f"  {r['N']:2d} fish {r['kind']:7} #{r['rec']}: G motion {g1['motion']:+.3f}/{g3['motion']:+.3f}"
                     f"  shape {g1['shape']:+.3f}/{g3['shape']:+.3f}"
                     f"  | per variable 1 s " + " ".join(f"{x:+.2f}" for x in g1["per_variable"])
                     + f"  | lead {r['lead']['coef']:+.3f} (t {r['lead']['t']:+.1f}, std {r['lead']['std_coef']:+.3f})"
                     f"  | speed lead {r['speed_lead']:+.2f} (t {r['speed_lead_t']:+.1f})")
    lines.append("  (per-variable order: " + ", ".join(Z_NAMES) + ")")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    tasks = [(39, r, "real") for r in (0, 1)] + [(8, r, k) for r in range(6) for k in ("real", "virtual")]
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run, tasks))
    res = {"runs": runs, "horizons": list(HORIZONS)}
    (out_dir / "e50_school_configuration_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e50_figures import make_figures
    make_figures(res, out_dir / "e50_figures", text)
    return text


if __name__ == "__main__":
    main()
