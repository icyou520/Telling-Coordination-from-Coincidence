"""Experiment 52: what inside the school tells it where to turn? A learned influence map.

e51 found the school closed for its shape and motion, but not for its next
turn: the internal configuration adds 6-21% to 1 s forecasts of the school's
heading change, and the two leadership features carry only 13-56% of that.
e52 stops choosing features. It learns an influence function f over each
fish's state, and the school's turning variable is the mean over fish:

    C(t) = mean_i  sum_c c(s_i) f_c(x_i, y_i)

x, y = the fish's place in the school's frame (x ahead, y to the left, both in
units of the school's spread); channels c = 1 (place), sin phi and cos phi
(heading relative to the school), omega (the fish's own turning in the last
0.1 s, relative to the school's), u (relative speed) and u sin phi. Each f_c is
a sum of Gaussian bumps (x centres -1.5..1.5, y centres 0, 0.75, 1.5, width 0.5)
made mirror-symmetric: a left turn of a mirrored school is a right turn, so f_c
is even in y for odd channels (sin phi, omega, u sin phi) and odd for even ones.
That gives 75 features, linear in the weights, so the fit is e51's validated
cross-validated ridge and the weights draw as maps.

Data and test: as e51 (six 8-fish recordings, real and virtual, 0.1 s steps,
first 70% fitted, last 30% measured; block bootstrap; school-state block =
e51's). Target: the school's heading change over h = 1 s (3 s exploratory).
Models: block; + conf (e51's configuration); + infl (the influence features);
conf + infl. Null: the influence features of a recording of different length
(e51 showed equal-length virtual recordings share seams). Calibration: plant
a 5% share of the block's error built from the part of the sin-phi feature at
(0.75, 0) the block cannot express.

Predictions, fixed before running (1 s horizon, 8-fish):
  D0  The test still works here: planted signal raises G by >= 0.025 with lower
      bound > 0 in >= 10/12 groups, AND null |G| <= 0.01 in >= 10/12.
  D1  Discovery at least matches the hand-made configuration:
      G(infl) >= G(conf) in >= 5 of 6 real schools.
  D2  It finds information beyond the configuration: 1 - MSE(conf+infl) /
      MSE(conf) >= 0.02 with lower bound > 0 in >= 5 of 6 real schools.
  D3  Front fish lead: the learned sin-phi map is positive at the front
      (x = 1, y = 0) and larger there than at the back (x = -1, y = 0), in >= 5
      of 6 real schools.
  D4  It is interaction, not individual persistence: G(infl) real > virtual in
      >= 5 of 6 pairs.
Exploratory: which channel carries it (block + one channel group at a time);
3 s; the two 39-fish recordings; the maps themselves.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e49_school_layer import LAGS, school_state
from experiments.e50_school_configuration import _wrap, configuration, school_map
from experiments.e51_trusted_closure import PLANT, Ridge, _tame, gap_ci

SEED = 52
HORIZONS = (10, 30)
BX, BY, WIDTH = (-1.5, -0.75, 0.0, 0.75, 1.5), (0.0, 0.75, 1.5), 0.5
CHANNELS = (("place", +1), ("sin", -1), ("cos", +1), ("turn", -1), ("speed", +1), ("speed_sin", -1))
GROUPS = {"heading": ("sin", "cos"), "turning": ("turn",), "place": ("place",), "speed": ("speed", "speed_sin")}
GRID = np.linspace(-2.5, 2.5, 41)


def bumps():
    """(channel, parity, a, b) for every influence feature; even channels drop b = 0 (identically zero)."""
    return [(c, par, a, b) for c, par in CHANNELS for a in BX for b in BY if not (par > 0 and b == 0)]


def _bump(x, y, a, b):
    return np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * WIDTH ** 2))


def fish_frame(p, th, sp):
    """Each fish in the school's frame: x ahead, y left (units of spread), phi, omega, u."""
    big_th = np.arctan2(np.sin(th).mean(1), np.cos(th).mean(1))
    r = p - p.mean(1)[:, None, :]
    spread = np.maximum(np.linalg.norm(r, axis=-1).mean(1), 1e-6)[:, None]
    ca, sa = np.cos(big_th)[:, None], np.sin(big_th)[:, None]
    x = (r[..., 0] * ca + r[..., 1] * sa) / spread
    y = (-r[..., 0] * sa + r[..., 1] * ca) / spread
    phi = _wrap(th - big_th[:, None])
    omega = _wrap(_wrap(th - np.roll(th, 1, axis=0)) - _wrap(big_th - np.roll(big_th, 1))[:, None])
    u = sp / np.maximum(sp.mean(1, keepdims=True), 1e-6) - 1
    return x, y, phi, omega, u


def _tame_fish(v, ref):
    med = np.median(v[ref])
    mad = np.median(np.abs(v[ref] - med)) or 1.0
    return np.arcsinh((v - med) / mad)


def influence(p, th, sp, ref):
    """The 75 mirror-symmetric influence features, shape (T, F)."""
    x, y, phi, omega, u = fish_frame(p, th, sp)
    om, ut = _tame_fish(omega, ref), _tame_fish(u, ref)
    ch = {"place": np.ones_like(x), "sin": np.sin(phi), "cos": np.cos(phi), "turn": om, "speed": ut,
          "speed_sin": ut * np.sin(phi)}
    cols = [(ch[c] * (_bump(x, y, a, b) - par * _bump(x, -y, a, b))).mean(1) for c, par, a, b in bumps()]
    return np.column_stack(cols), x, y


def channel_maps(w):
    """Learned f_c on GRID x GRID for each channel, from the raw influence weights."""
    gx, gy = np.meshgrid(GRID, GRID)
    maps = {c: np.zeros_like(gx) for c, _ in CHANNELS}
    for wk, (c, par, a, b) in zip(w, bumps()):
        maps[c] += wk * (_bump(gx, gy, a, b) - par * _bump(gx, -gy, a, b))
    return maps


class MapRidge(Ridge):
    def fit_coef(self, y_train, w):
        """As fit, plus raw-unit coefficients of the full-training fit."""
        pred, lam, s = self.fit(y_train, w)
        mu, sd, eig, n = self.full
        ev, v = eig[s]
        ym = y_train.mean(0)
        r = ((self.x[self.train] - mu) / sd).T @ (y_train - ym)
        proj = v.T @ (self.d[s][:, None] * r)
        coef = self.d[s][:, None] * (v @ (proj / (np.clip(ev, 0, None) + lam * n)[:, None]))
        return pred, lam, s, coef / sd[:, None]


def _density(x, y, rows):
    h, _, _ = np.histogram2d(y[rows].ravel(), x[rows].ravel(), bins=len(GRID),
                             range=[[GRID[0], GRID[-1]], [GRID[0], GRID[-1]]])
    return (h / h.sum()).tolist()


def _run(args):
    n, rec, kind, null_rec = args
    from experiments.e31_real_fish import kinematics, virtual
    from experiments.fish_data import load
    rng = np.random.default_rng(SEED * 1000 + 10 * n + rec + (500 if kind == "virtual" else 0))
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    seam = (sp > 100.0).any(1)
    sp = np.where(sp > 100.0, 0.0, sp)
    T = len(th)
    cut, hmax = int(0.7 * T), max(HORIZONS)
    bad = np.zeros(T, bool)
    for s in np.nonzero(seam)[0]:
        bad[max(0, s - hmax - 1):s + LAGS + 2] = True
    rows = np.arange(LAGS + 1, T - hmax)
    rows = rows[~bad[rows]]
    train, test = rows[rows + hmax < cut], rows[rows >= cut]
    rows = np.concatenate([train, test])

    z = school_state(p, th, sp)
    conf, lead, big_th = configuration(p, th, sp)
    conf = _tame(np.column_stack([conf, lead]), train)
    infl, fx, fy = influence(p, th, sp, train)
    rates = np.column_stack([np.roll(_wrap(big_th - np.roll(big_th, 1)), j) for j in range(5)])
    iu = np.triu_indices(z.shape[1])
    lagged = np.concatenate([np.roll(z, j, axis=0) for j in range(LAGS)], 1)
    quad = (z[:, :, None] * z[:, None, :])[:, iu[0], iu[1]]
    block = np.concatenate([lagged, quad, school_map(z), rates], 1)
    with_conf = np.concatenate([block, conf, conf ** 2], 1)
    nb, nc = block.shape[1], with_conf.shape[1]
    names = bumps()
    models = {"conf": (with_conf, nb), "infl": (np.concatenate([block, infl], 1), nb),
              "conf_infl": (np.concatenate([with_conf, infl], 1), nc)}
    for g, chans in GROUPS.items():
        idx = [j for j, (c, *_) in enumerate(names) if c in chans]
        models["only_" + g] = (np.concatenate([block, infl[:, idx]], 1), nb)
    if null_rec is not None:
        po, tho, spo = kinematics(load(n, null_rec) if kind == "real" else virtual(load(n, null_rec)))
        spo = np.where(spo > 100.0, 0.0, spo)
        null, _, _ = influence(po, tho, spo, np.arange(len(tho)))
        models["null"] = (np.concatenate([block, np.resize(null, infl.shape)], 1), nb)
    ones_block = np.column_stack([np.ones(T), block])
    j0 = names.index(("sin", -1, 0.75, 0.0))
    xi = infl[:, j0] - ones_block @ np.linalg.lstsq(ones_block[rows], infl[rows, j0], rcond=None)[0]

    out = {"N": n, "rec": rec, "kind": kind, "null_rec": null_rec, "density": _density(fx, fy, train), "h": {}}
    for h in HORIZONS:
        y = _wrap(big_th[rows + h] - big_th[rows])
        pos_tr, pos_te = np.arange(len(train)), np.arange(len(train), len(rows))
        y_tr, y_te = y[pos_tr, None], y[pos_te]
        one = np.ones(1)
        base = Ridge(block, train, test, h + LAGS)
        pb, lam_b, _ = base.fit(y_tr, one)
        pb = pb[:, 0]
        res = {"lam_block": lam_b}
        preds = {}
        for k, (x, nblk) in models.items():
            m = MapRidge(x, train, test, h + LAGS, nblk)
            pr, lam, s, coef = m.fit_coef(y_tr, one)
            preds[k] = pr[:, 0]
            res[k] = gap_ci((y_te - preds[k]) ** 2, (y_te - pb) ** 2, rng) | {"lam": lam, "scale": s}
            if k == "infl" and h == 10:
                maps = channel_maps(coef[nb:, 0])
                out["maps"] = {c: v.tolist() for c, v in maps.items()}
                i_f, i_b = np.argmin(np.abs(GRID - 1)), np.argmin(np.abs(GRID + 1))
                i_0 = np.argmin(np.abs(GRID))
                out["front_back"] = [float(maps["sin"][i_0, i_f]), float(maps["sin"][i_0, i_b])]
            if k == "infl":
                pm = m
        res["beyond_conf"] = gap_ci((y_te - preds["conf_infl"]) ** 2, (y_te - preds["conf"]) ** 2, rng)
        if n == 8:
            m_b = ((y_te - pb) ** 2).mean()
            kappa = np.sqrt(PLANT / (1 - PLANT) * m_b / xi[test].var())
            yp = y + kappa * xi[rows]
            fb = base.fit(yp[pos_tr, None], one)[0][:, 0]
            fi = pm.fit(yp[pos_tr, None], one)[0][:, 0]
            res["planted"] = {"before": res["infl"],
                              "after": gap_ci((yp[pos_te] - fi) ** 2, (yp[pos_te] - fb) ** 2, rng),
                              "kappa": float(kappa)}
        out["h"][str(h)] = res
    return out


def _sel(res, n, kind):
    return sorted([r for r in res["runs"] if r["N"] == n and r["kind"] == kind], key=lambda r: r["rec"])


def predictions(res):
    real, virt = _sel(res, 8, "real"), _sel(res, 8, "virtual")
    g8 = [r["h"]["10"] for r in real + virt]
    cal = sum(g["planted"]["after"]["G"] - g["planted"]["before"]["G"] >= 0.025 and g["planted"]["after"]["lo"] > 0
              for g in g8)
    nul = sum(abs(g["null"]["G"]) <= 0.01 for g in g8)
    rg = [r["h"]["10"] for r in real]
    d1 = sum(g["infl"]["G"] >= g["conf"]["G"] for g in rg)
    d2 = sum(g["beyond_conf"]["G"] >= 0.02 and g["beyond_conf"]["lo"] > 0 for g in rg)
    d3 = sum(f > 0 and f > b for f, b in (r["front_back"] for r in real))
    d4 = sum(r["h"]["10"]["infl"]["G"] > v["h"]["10"]["infl"]["G"] for r, v in zip(real, virt))
    fmt = lambda gs, k: ", ".join(f"{g[k]['G']:+.3f}" for g in gs)  # noqa: E731
    return [
        ("D0", "the test works: planted rise >= 0.025, lower > 0 (>= 10/12) and null |G| <= 0.01 (>= 10/12)",
         cal >= 10 and nul >= 10, f"planted {cal}/12, null {nul}/12"),
        ("D1", "discovery matches the configuration: G(infl) >= G(conf) (>= 5/6 real)", d1 >= 5,
         f"{d1}/6; infl {fmt(rg, 'infl')} vs conf {fmt(rg, 'conf')}"),
        ("D2", "beyond the configuration: gain >= 0.02, lower > 0 (>= 5/6 real)", d2 >= 5,
         f"{d2}/6; {fmt(rg, 'beyond_conf')}"),
        ("D3", "front fish lead: sin-phi map > 0 at front and front > back (>= 5/6 real)", d3 >= 5,
         f"{d3}/6; front/back " + ", ".join(f"{f:+.3f}/{b:+.3f}" for f, b in (r['front_back'] for r in real))),
        ("D4", "interaction, not persistence: G(infl) real > virtual (>= 5/6 pairs)", d4 >= 5,
         f"{d4}/6; virtual {fmt([v['h']['10'] for v in virt], 'infl')}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 52: a learned influence map for the school's turn", ""]
    keys = ["conf", "infl", "conf_infl", "beyond_conf"] + ["only_" + g for g in GROUPS]
    for r in sorted(res["runs"], key=lambda r: (r["N"], r["kind"], r["rec"])):
        lines.append(f"  {r['N']:2d} fish {r['kind']:7} #{r['rec']}" + (f" (null from #{r['null_rec']})"
                                                                          if r["null_rec"] is not None else ""))
        for h in res["horizons"]:
            g = r["h"][str(h)]
            lines.append(f"    {h / 10:.0f} s  " + "  ".join(
                f"{k} {g[k]['G']:+.3f} [{g[k]['lo']:+.3f},{g[k]['hi']:+.3f}]" for k in keys[:4]))
            extra = "  ".join(f"{k[5:]} {g[k]['G']:+.3f}" for k in keys[4:])
            if "null" in g:
                extra += (f" | null {g['null']['G']:+.3f} | planted {g['planted']['before']['G']:+.3f} -> "
                          f"{g['planted']['after']['G']:+.3f} (lo {g['planted']['after']['lo']:+.3f})")
            lines.append("          one channel group: " + extra)
        f, b = r["front_back"]
        lines.append(f"          sin-phi map: front {f:+.4f}, back {b:+.4f}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def null_recordings(n, count):
    """For each recording, the next one (cyclically) whose length differs."""
    from experiments.fish_data import load
    lens = [len(load(n, r)) for r in range(count)]
    out = []
    for r in range(count):
        k = next(k for k in range(1, count) if lens[(r + k) % count] != lens[r])
        out.append((r + k) % count)
    return out


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    nulls = null_recordings(8, 6)
    tasks = [(39, r, "real", None) for r in (0, 1)]
    tasks += [(8, r, k, nulls[r]) for r in range(6) for k in ("real", "virtual")]
    if quick:
        tasks = [(8, 0, "real", nulls[0]), (8, 0, "virtual", nulls[0])]
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run, tasks))
    res = {"runs": runs, "horizons": list(HORIZONS)}
    (out_dir / "e52_influence_map_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e52_figures import make_figures
    make_figures(res, out_dir / "e52_figures", text)
    return text


if __name__ == "__main__":
    main()
