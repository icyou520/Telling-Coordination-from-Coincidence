"""Experiment 51: a closure test we can trust, then the school's closure again.

e48 and e50 showed that "closed" depends on the forecaster. A weak one (e48)
calls everything closed; an overfitting one (e50: 40 extra features by least
squares, G down to -1.2) calls everything worse. e51 replaces it with ridge
regression chosen by blocked cross-validation inside the training data (5
contiguous folds, a gap of h + 5 steps around each). The extended model has two
knobs, the penalty and a scale on the added features (0, 0.1, 0.3, 1, 3 after
standardizing; 0 = the block model), so it contains the block model and can
decline useless features. Configuration features are tamed by
asinh((x - median) / MAD), because x'^2 sin phi etc. have tails 100x their
spread. The test must pass two checks on itself before its answer about real
schools counts.

Development note: the method (not the predictions K1-K6, written first) was
fixed on recording 0 only, real and virtual, after the first version failed its
own checks: a T/2-rolled null still carried information in virtual groups
(virtual() shifts fish i by i*T/8, so the roll re-pairs fish from the same real
moments), the heavy-tailed planted feature was invisible in the test rows, and
one shared penalty made useless features cost up to 4%.

Data: the six 8-fish recordings (real and virtual, as in e48-e50), 0.1 s steps,
fitted on the first 70%, measured on the last 30%. Uncertainty: moving-block
bootstrap of the test rows (blocks of 30 s, 400 resamples), 95% interval.

Targets (changes over h = 1 s and 3 s): the school's heading change dTheta;
motion = (centroid, mean heading vector, mean speed); shape = (polarization,
spread, milling); all = the 8 school variables. Multi-variable G is whitened by
the covariance of the change (e47).
Features:
  block   = e50's school-state block (Z_t..Z_{t-4}, quadratic terms of Z_t,
            the school's place-and-heading map) + the last 5 turning rates
  + conf  = block + e50's 20 configuration features, leadership L, and squares
  + null  = block + the configuration features of the next recording (same
            kind, tiled to length): same kind of features, no information
G = 1 - MSE(block + extra) / MSE(block), each model with its own penalty.

Calibration (known answer): in every group, add to the target a planted
signal kappa * xi, xi the part of a configuration feature the block cannot
express linearly (x'^2 sin phi for dTheta; relative speed x front position
u x' for spread change), with kappa^2 var_test(xi) = 0.05/0.95 * MSE_block:
the test should find G rise by about 0.05 (1 - G).

Predictions, fixed before running (1 s horizon):
  K1  Calibration: the planted signal raises G by >= 0.025 and its 95% lower
      bound is > 0, for both planted targets, in >= 10 of 12 8-fish groups.
  K2  Null features: -0.01 <= G <= 0.01 for dTheta, motion and shape in
      >= 10 of 12 8-fish groups.
  K3  Leadership through the general test: G(dTheta) >= 0.03 with lower
      bound > 0 in >= 5 of 6 real schools.
  K4  Configuration informs the school's shape: G(shape) >= 0.05 with lower
      bound > 0 in >= 5 of 6 real schools.
  K5  Configuration informs the school's motion: G(motion) lower bound > 0
      in >= 5 of 6 real schools.
  K6  The school is more self-contained than an average of independent fish:
      G(all) real < G(all) virtual in >= 5 of 6 pairs.
Exploratory: 3 s horizon; G(dTheta) with x' sin phi and L removed from the
configuration (how much of the gain is leadership); e50's unregularized fit
on the same features; the two 39-fish recordings (real only).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e48_fish_layers import _lstsq
from experiments.e49_school_layer import LAGS, school_state
from experiments.e50_school_configuration import CONF_NAMES, _wrap, configuration, school_map

SEED = 51
HORIZONS = (10, 30)
LAMBDAS = np.logspace(-5, 2, 15)          # penalty per row, on standardized features
SCALES = (0.0, 0.1, 0.3, 1.0, 3.0)       # extra features' scale after standardizing (0 = block model)
FOLDS = 5
BOOT, BLOCK = 400, 300
PLANT = 0.05
TARGETS = {"turn": (0,), "motion": (1, 2, 3, 4, 5), "shape": (6, 7, 8), "all": (1, 2, 3, 4, 5, 6, 7, 8)}
LEAD_FEATS = (CONF_NAMES.index("x_sin"), len(CONF_NAMES))   # x' sin phi and L


class Ridge:
    """Ridge on standardized features; penalty chosen by blocked cross-validation."""

    def __init__(self, x, train, test, gap, n_block=None):
        self.x, self.train, self.test = x, train, test
        self.scales = (1.0,) if n_block is None else SCALES
        self.d = {s: np.where(np.arange(x.shape[1]) < (n_block or x.shape[1]), 1.0, s) for s in self.scales}
        self.cv = []
        for f in np.array_split(np.arange(len(train)), FOLDS):
            keep = np.ones(len(train), bool)
            keep[max(0, f[0] - gap):f[-1] + gap + 1] = False
            self.cv.append((self._gram(train[keep]), keep, f))
        self.full = self._gram(train)

    def _gram(self, rows):
        a = self.x[rows]
        mu, sd = a.mean(0), a.std(0)
        sd[sd == 0] = 1.0
        a = (a - mu) / sd
        g = a.T @ a
        eig = {s: np.linalg.eigh(self.d[s][:, None] * g * self.d[s][None, :]) for s in self.scales}
        return mu, sd, eig, len(rows)

    def _paths(self, gram, fit_rows, y, pred_rows):
        """Predictions on pred_rows for every (scale, penalty)."""
        mu, sd, eig, n = gram
        ym = y.mean(0)
        r = ((self.x[fit_rows] - mu) / sd).T @ (y - ym)
        b = (self.x[pred_rows] - mu) / sd
        out = {}
        for s in self.scales:
            ev, v = eig[s]
            proj = v.T @ (self.d[s][:, None] * r)
            for k, lam in enumerate(LAMBDAS):
                coef = self.d[s][:, None] * (v @ (proj / (np.clip(ev, 0, None) + lam * n)[:, None]))
                out[s, k] = b @ coef + ym
        return out

    def fit(self, y_train, w):
        """Test-row predictions, chosen penalty and extra-feature scale; y_train: one column per target."""
        score = {}
        for gram, keep, f in self.cv:
            for key, pred in self._paths(gram, self.train[keep], y_train[keep], self.train[f]).items():
                score[key] = score.get(key, 0.0) + (((y_train[f] - pred) ** 2) * w).sum()
        s, k = min(score, key=score.get)
        pred = self._paths(self.full, self.train, y_train, self.test)[s, k]
        return pred, float(LAMBDAS[k]), float(s)


def gap_ci(q_extra, q_block, rng):
    """G = 1 - sum(q_extra)/sum(q_block) over test rows, with a moving-block bootstrap interval."""
    g = 1 - q_extra.sum() / q_block.sum()
    n = len(q_block)
    nb = max(1, n // BLOCK)
    starts = rng.integers(0, n - BLOCK + 1, size=(BOOT, nb))
    idx = (starts[..., None] + np.arange(BLOCK)).reshape(BOOT, -1)
    boot = 1 - q_extra[idx].sum(1) / q_block[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"G": float(g), "lo": float(lo), "hi": float(hi)}


def _tame(x, ref):
    """asinh((x - median) / MAD) per column, median and MAD from the reference rows: monotone, light tails."""
    med = np.median(x[ref], 0)
    mad = np.median(np.abs(x[ref] - med), 0)
    mad[mad == 0] = 1.0
    return np.arcsinh((x - med) / mad)


def _run(args):
    n, rec, kind = args
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
    other = load(n, (rec + 1) % 6) if n == 8 else None
    if other is not None:
        po, tho, spo = kinematics(other if kind == "real" else virtual(other))
        co, lo, _ = configuration(po, tho, np.where(spo > 100.0, 0.0, spo))
        null = np.resize(_tame(np.column_stack([co, lo]), np.arange(len(co))), conf.shape)
    rates = np.column_stack([np.roll(_wrap(big_th - np.roll(big_th, 1)), j) for j in range(5)])
    iu = np.triu_indices(z.shape[1])
    lagged = np.concatenate([np.roll(z, j, axis=0) for j in range(LAGS)], 1)
    quad = (z[:, :, None] * z[:, None, :])[:, iu[0], iu[1]]
    block = np.concatenate([lagged, quad, school_map(z), rates], 1)
    no_lead = np.delete(conf, LEAD_FEATS, axis=1)
    feats = {"block": block,
             "conf": np.concatenate([block, conf, conf ** 2], 1),
             "conf_no_lead": np.concatenate([block, no_lead, no_lead ** 2], 1)}
    if n == 8:
        feats["null"] = np.concatenate([block, null, null ** 2], 1)
    ones_block = np.column_stack([np.ones(T), block])

    def unique(v):
        """The part of a configuration feature the block cannot express linearly."""
        return v - ones_block @ np.linalg.lstsq(ones_block[rows], v[rows], rcond=None)[0]

    xi_turn = unique(conf[:, CONF_NAMES.index("x2_sin")])
    xi_spread = unique(conf[:, CONF_NAMES.index("u_x")])

    out = {"N": n, "rec": rec, "kind": kind, "h": {}}
    for h in HORIZONS:
        d = np.column_stack([_wrap(big_th[rows + h] - big_th[rows]), z[rows + h] - z[rows]])
        pos_tr, pos_te = np.arange(len(train)), np.arange(len(train), len(rows))
        assert (rows[pos_te] == test).all()
        d_tr, d_te = d[pos_tr], d[pos_te]
        w = 1 / d_tr.var(0)
        models = {k: Ridge(f, train, test, h + LAGS, None if k == "block" else block.shape[1])
                  for k, f in feats.items()}
        per = {}                                                    # penalty and scale per target group
        for tname, idx in TARGETS.items():
            i = list(idx)
            per[tname] = {k: m.fit(d_tr[:, i], w[i]) for k, m in models.items()}
        res = {}
        for tname, idx in TARGETS.items():
            i = list(idx)
            vinv = np.linalg.inv(np.atleast_2d(np.cov(d_te[:, i].T)))
            q = {k: np.einsum("ti,ij,tj->t", d_te[:, i] - pr[0], vinv, d_te[:, i] - pr[0])
                 for k, pr in per[tname].items()}
            res[tname] = {k: gap_ci(q[k], q["block"], rng) | {"lam": per[tname][k][1], "scale": per[tname][k][2]}
                          for k in q if k != "block"}
            res[tname]["lam_block"] = per[tname]["block"][1]
            if n == 8:
                xb, xc = feats["block"], feats["conf"]
                e_b = d_te[:, i] - _lstsq(np.column_stack([np.ones(len(rows)), xb[rows]])[pos_tr], d_tr[:, i],
                                          np.column_stack([np.ones(len(rows)), xb[rows]])[pos_te])
                e_c = d_te[:, i] - _lstsq(np.column_stack([np.ones(len(rows)), xc[rows]])[pos_tr], d_tr[:, i],
                                          np.column_stack([np.ones(len(rows)), xc[rows]])[pos_te])
                qb = np.einsum("ti,ij,tj->t", e_b, vinv, e_b)
                qc = np.einsum("ti,ij,tj->t", e_c, vinv, e_c)
                res[tname]["ols_conf"] = float(1 - qc.sum() / qb.sum())
        if n == 8:
            for tname, col, xi in (("turn_planted", 0, xi_turn), ("spread_planted", 7, xi_spread)):
                y = d[:, col]
                base = {k: models[k].fit(y[pos_tr, None], np.ones(1))[0][:, 0] for k in ("block", "conf")}
                m_b = ((y[pos_te] - base["block"]) ** 2).mean()
                kappa = np.sqrt(PLANT / (1 - PLANT) * m_b / xi[test].var())
                yp = y + kappa * xi[rows]
                fit = {k: models[k].fit(yp[pos_tr, None], np.ones(1))[0][:, 0] for k in ("block", "conf")}
                q0 = {k: (y[pos_te] - base[k]) ** 2 for k in base}
                q1 = {k: (yp[pos_te] - fit[k]) ** 2 for k in fit}
                res[tname] = {"before": gap_ci(q0["conf"], q0["block"], rng),
                              "after": gap_ci(q1["conf"], q1["block"], rng), "kappa": float(kappa)}
        out["h"][str(h)] = res
    return out


def _sel(res, n, kind):
    return sorted([r for r in res["runs"] if r["N"] == n and r["kind"] == kind], key=lambda r: r["rec"])


def predictions(res):
    real, virt = _sel(res, 8, "real"), _sel(res, 8, "virtual")
    g8 = [r["h"]["10"] for r in real + virt]
    k1 = {t: sum(g[t]["after"]["G"] - g[t]["before"]["G"] >= 0.025 and g[t]["after"]["lo"] > 0 for g in g8)
          for t in ("turn_planted", "spread_planted")}
    k2 = {t: sum(-0.01 <= g[t]["null"]["G"] <= 0.01 for g in g8) for t in ("turn", "motion", "shape")}
    rg = [r["h"]["10"] for r in real]
    k3 = sum(g["turn"]["conf"]["G"] >= 0.03 and g["turn"]["conf"]["lo"] > 0 for g in rg)
    k4 = sum(g["shape"]["conf"]["G"] >= 0.05 and g["shape"]["conf"]["lo"] > 0 for g in rg)
    k5 = sum(g["motion"]["conf"]["lo"] > 0 for g in rg)
    k6 = sum(r["h"]["10"]["all"]["conf"]["G"] < v["h"]["10"]["all"]["conf"]["G"] for r, v in zip(real, virt))
    fmt = lambda gs, t, k="conf": ", ".join(f"{g[t][k]['G']:+.3f}" for g in gs)  # noqa: E731
    return [
        ("K1", "calibration: planted 5% signal raises G >= 0.025, lower bound > 0 (>= 10/12, both targets)",
         min(k1.values()) >= 10, f"turn {k1['turn_planted']}/12, spread {k1['spread_planted']}/12"),
        ("K2", "null features: |G| <= 0.01 for turn, motion, shape (>= 10/12 each)", min(k2.values()) >= 10,
         f"turn {k2['turn']}/12, motion {k2['motion']}/12, shape {k2['shape']}/12"),
        ("K3", "leadership via the general test: G(turn) >= 0.03, lower > 0 (>= 5/6 real)", k3 >= 5,
         f"{k3}/6; G {fmt(rg, 'turn')}"),
        ("K4", "configuration informs shape: G(shape) >= 0.05, lower > 0 (>= 5/6 real)", k4 >= 5,
         f"{k4}/6; G {fmt(rg, 'shape')}"),
        ("K5", "configuration informs motion: G(motion) lower > 0 (>= 5/6 real)", k5 >= 5,
         f"{k5}/6; G {fmt(rg, 'motion')}"),
        ("K6", "real more self-contained than virtual: G(all) real < virtual (>= 5/6 pairs)", k6 >= 5,
         f"{k6}/6; real {fmt(rg, 'all')} vs virtual {fmt([v['h']['10'] for v in virt], 'all')}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 51: a closure test we can trust (cross-validated ridge, calibrated, null-checked)", ""]
    for r in sorted(res["runs"], key=lambda r: (r["N"], r["kind"], r["rec"])):
        lines.append(f"  {r['N']:2d} fish {r['kind']:7} #{r['rec']}")
        for h in res["horizons"]:
            g = r["h"][str(h)]
            cell = lambda t, k: f"{g[t][k]['G']:+.3f} [{g[t][k]['lo']:+.3f}, {g[t][k]['hi']:+.3f}]"  # noqa: E731
            row = f"    {h / 10:.0f} s  conf: " + "  ".join(f"{t} {cell(t, 'conf')}" for t in TARGETS)
            lines.append(row)
            lines.append(f"          conf w/o leadership: turn {cell('turn', 'conf_no_lead')}"
                         + (f" | null: " + "  ".join(f"{t} {g[t]['null']['G']:+.3f}" for t in TARGETS)
                            + " | e50-style unregularized: " + "  ".join(f"{t} {g[t]['ols_conf']:+.3f}"
                                                                         for t in TARGETS) if r["N"] == 8 else ""))
            if r["N"] == 8:
                lines.append("          planted: " + "  ".join(
                    f"{t} {g[t]['before']['G']:+.3f} -> {g[t]['after']['G']:+.3f} (lo {g[t]['after']['lo']:+.3f})"
                    for t in ("turn_planted", "spread_planted"))
                    + "  | penalty conf/block (turn) "
                    f"{g['turn']['conf']['lam']:.0e}/{g['turn']['lam_block']:.0e}, extra scale conf/null "
                    + " ".join(f"{t}:{g[t]['conf']['scale']:g}/{g[t]['null']['scale']:g}" for t in TARGETS))
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
    if quick:
        tasks = [(8, 0, "real"), (8, 0, "virtual")]
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run, tasks))
    res = {"runs": runs, "horizons": list(HORIZONS)}
    (out_dir / "e51_trusted_closure_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e51_figures import make_figures
    make_figures(res, out_dir / "e51_figures", text)
    return text


if __name__ == "__main__":
    main()
