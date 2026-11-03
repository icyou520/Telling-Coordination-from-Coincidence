"""Experiment 63: is a brain area's population state a closed level?

e62 measured how much coordination shapes an area's activity (the interaction share). e63 asks the
closure question of the same areas: does the area's summary state, with its recorded causes, forecast
itself as well as the full population does? If yes, the summary is a level of description with its
own law; if the rest of the population (or other areas) adds forecast power, the level is open.

Data: the 221 areas of e62 (Steinmetz et al. 2019; >= 30 neurons firing >= 0.5 Hz). Each neuron's
counts in a window are square-rooted and standardized (training rows); principal components from the
50 ms windows of training rows. Forecast rows: every second bin from 19 to 229 of each trial.
Summary (block): the top 3 components in the last 5, 5-10, 10-20 bins, the current ones squared, the
area's total count in the same windows, and e62's E1 recorded causes (stimulus, go cue, feedback,
wheel, pupil, face, licks, session time).
Targets, h = 5, 10, 20 bins ahead (50, 100, 200 ms; the 50 ms window ending at t + h):
  state  the top 3 components (G whitened by their covariance, as e47);
  rate   the total count.
Extra features:
  internal  components 4-30 in the last 5 and 5-10 bins (the area's state beyond its summary);
  other     the log activity of every other area with >= 10 neurons, last 5 and 5-20 bins (e62's E2);
  null      the internal features of another trial (trials rolled by half, within training and
            within test trials): same kind, no information.
Test: e51's ridge (penalty and extra-feature scale chosen by blocked cross-validation inside the
training trials). G = 1 - MSE(block + extra) / MSE(block). Training and test trials as e62 (blocks
of 10, every third held out); 90% interval from 400 bootstrap resamples of test trials.
Planted check: add kappa * xi to the first component at 50 ms, xi = the part of the first internal
feature the block cannot express linearly, sized to explain 5% of the block's error.
Periods (same fitted models, test rows split by where the target window lies): pre-stimulus (ends
before onset), response (0-0.5 s after onset), late (after 0.5 s).

Predictions, fixed before running:
  C1  Calibration: the planted signal raises G by >= 0.025 with lower bound > 0 in >= 90% of areas.
  C2  Null: |G_null(state)| <= 0.01 at 50 ms in >= 90% of areas.
  C3  Areas are open inside at 50 ms: G_internal(state) >= 0.02 with lower bound > 0 in >= 50%.
  C4  Closure grows with horizon: median G_internal(state) at 200 ms < at 50 ms.
  C5  Areas are open to other areas: G_other(state) lower bound > 0 at 50 ms in >= 50%.
  C6  Interaction makes levels open: Spearman(e62 S_proj(E2), G_internal(rate) at 50 ms) >= 0.3.
  C7  The stimulus opens the level: median G_internal(state) at 50 ms is higher in the response period
      than before the stimulus.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e51_trusted_closure import Ridge
from experiments.e61_neural_share import _cs, split
from experiments.e62_neural_share_repaired import MIN_NEURONS, MIN_RATE, env_areas, env_task, env_stim

SEED = 63
NB = 250
STIM = 50
HORIZONS = (5, 10, 20)
TS = np.arange(19, NB - 20, 2)
HIST = ((0, 5), (5, 10), (10, 20))
K = 3
N_INT = 27
N_BOOT = 400
PLANT = 0.05
PERIODS = ("pre", "response", "late")


def _win(c, a, b, t=TS):
    """Sums of bins (t-b, t-a] from cumulative sums c (..., bins + 1)."""
    return c[..., t + 1 - a] - c[..., t + 1 - b]


def _env(td):
    """e62's E1 at this experiment's rows."""
    import experiments.e62_neural_share_repaired as e62
    saved = e62.TS
    e62.TS = TS
    try:
        return np.concatenate([env_stim(td["cl"], td["cr"]), env_task(td)], -1)
    finally:
        e62.TS = saved


def _env_other(others):
    import experiments.e62_neural_share_repaired as e62
    saved = e62.TS
    e62.TS = TS
    try:
        return env_areas(others)
    finally:
        e62.TS = saved


def features(x, train_tr):
    """Component windows, targets and rates. x: (neurons, trials, bins)."""
    n, nt, _ = x.shape
    c = _cs(x)
    w5 = np.sqrt(_win(c, 0, 5))                                     # (n, nt, rows)
    tr = w5[:, train_tr].reshape(n, -1)
    mu, sd = tr.mean(1), tr.std(1)
    sd[sd == 0] = 1.0
    z = lambda w: (w - mu[:, None, None]) / sd[:, None, None]  # noqa: E731
    ev, vec = np.linalg.eigh(np.cov(z(w5[:, train_tr]).reshape(n, -1)))
    load = vec[:, ::-1][:, :K + N_INT]

    def comps(a, b, t=TS):
        w = np.sqrt(_win(c, a, b, t) * 5 / (b - a))
        return np.einsum("nk,ntr->trk", load, z(w))

    hist = [comps(a, b) for a, b in HIST]
    targets = {h: comps(0, 5, TS + h) for h in HORIZONS}
    ctot = _cs(x.sum(0))
    rate_hist = np.stack([_win(ctot, a, b) for a, b in HIST], -1)
    rate_t = {h: _win(ctot, 0, 5, TS + h) for h in HORIZONS}
    return hist, targets, rate_hist, rate_t


def _period(h):
    end = TS + h
    return np.select([end < STIM, end < STIM + 50], [0, 1], 2)


def _g(q_extra, q_block, nr, test_tr_count, rng, mask=None):
    qe, qb = q_extra.reshape(test_tr_count, nr), q_block.reshape(test_tr_count, nr)
    if mask is not None:
        qe, qb = qe[:, mask], qb[:, mask]
    se, sb = qe.sum(1), qb.sum(1)
    g = 1 - se.sum() / sb.sum()
    idx = rng.integers(0, len(se), (N_BOOT, len(se)))
    boot = 1 - se[idx].sum(1) / sb[idx].sum(1)
    lo, hi = np.quantile(boot, [0.05, 0.95])
    return {"G": float(g), "lo": float(lo), "hi": float(hi)}


def analyse_area(x, env, other, rng):
    n, nt, _ = x.shape
    nr = len(TS)
    train_tr, test_tr = split(nt)
    hist, targets, rate_hist, rate_t = features(x, train_tr)
    flat = lambda a: a.reshape(nt * nr, -1)  # noqa: E731
    block = np.concatenate([flat(np.concatenate([h[..., :K] for h in hist], -1)), flat(hist[0][..., :K] ** 2),
                            flat(rate_hist), flat(env)], 1)
    internal = flat(np.concatenate([hist[0][..., K:], hist[1][..., K:]], -1))
    roll = np.arange(nt)
    roll[train_tr] = np.roll(train_tr, len(train_tr) // 2)
    roll[test_tr] = np.roll(test_tr, len(test_tr) // 2)
    null = internal.reshape(nt, nr, -1)[roll].reshape(nt * nr, -1)
    extras = {"internal": internal, "null": null}
    if other is not None:
        extras["other"] = flat(other)
    rows_of = lambda trs: (trs[:, None] * nr + np.arange(nr)).ravel()  # noqa: E731
    train, test = rows_of(train_tr), rows_of(test_tr)
    nb = block.shape[1]
    models = {"block": Ridge(block, train, test, 10)}
    models.update({k: Ridge(np.concatenate([block, v], 1), train, test, 10, nb) for k, v in extras.items()})
    out = {}
    for h in HORIZONS:
        per = _period(h)
        tg = {"state": flat(targets[h][..., :K]), "rate": rate_t[h].reshape(-1, 1)}
        for name, y in tg.items():
            y_tr, y_te = y[train], y[test]
            w = 1 / y_tr.var(0)
            vinv = np.linalg.inv(np.atleast_2d(np.cov(y_te.T)))
            q = {}
            for k, m in models.items():
                e = y_te - m.fit(y_tr, w)[0]
                q[k] = np.einsum("ti,ij,tj->t", e, vinv, e)
            for k in extras:
                res = _g(q[k], q["block"], nr, len(test_tr), rng)
                res["periods"] = {p: _g(q[k], q["block"], nr, len(test_tr), rng, per == i)
                                  for i, p in enumerate(PERIODS) if (per == i).sum() >= 3}
                out[f"{name}_{h}_{k}"] = res
    y = targets[5][..., 0].ravel()
    ones = np.column_stack([np.ones(len(block)), block])
    xi = internal[:, 0] - ones @ np.linalg.lstsq(ones[train], internal[train, 0], rcond=None)[0]
    one = np.ones(1)
    pb = models["block"].fit(y[train, None], one)[0][:, 0]
    kappa = np.sqrt(PLANT / (1 - PLANT) * ((y[test] - pb) ** 2).mean() / xi[test].var())
    yp = y + kappa * xi
    fb = models["block"].fit(yp[train, None], one)[0][:, 0]
    fa = models["internal"].fit(yp[train, None], one)[0][:, 0]
    fb0 = pb
    fa0 = models["internal"].fit(y[train, None], one)[0][:, 0]
    out["planted"] = {"before": _g((y[test] - fa0) ** 2, (y[test] - fb0) ** 2, nr, len(test_tr), rng),
                      "after": _g((yp[test] - fa) ** 2, (yp[test] - fb) ** 2, nr, len(test_tr), rng)}
    return out


def _session_run(s):
    from experiments import steinmetz_data as sd
    d = sd.session(s)
    td = sd.trial_data(s)
    spk = d["spks"]
    names = d["brain_area"]
    rates = spk.mean((1, 2)) / 0.01
    env = _env(td)
    labels, counts = np.unique(names, return_counts=True)
    out = []
    for area in labels:
        sel = (names == area) & (rates >= MIN_RATE)
        if area == "root" or sel.sum() < MIN_NEURONS:
            continue
        others = [spk[names == a].sum(0) for a, c in zip(labels, counts) if a not in (area, "root") and c >= 10]
        other = _env_other(others) if others else None
        res = analyse_area(spk[sel].astype(np.int16), env, other,
                           np.random.default_rng([SEED, s, int(sel.sum())]))
        out.append({"session": s, "area": str(area), "n": int(sel.sum()), **res})
    return out


# ---------------------------------------------------------------- analysis

def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _e62(res_dir=Path("results")):
    path = res_dir / "e62_neural_share_repaired_raw.json"
    if not path.exists():
        return {}
    return {(r["session"], r["area"]): r["E2"]["S"] for r in json.loads(path.read_text())["real"]}


def predictions(res):
    r = res["real"]
    m = len(r)
    c1 = sum(x["planted"]["after"]["G"] - x["planted"]["before"]["G"] >= 0.025 and x["planted"]["after"]["lo"] > 0
             for x in r)
    c2 = sum(abs(x["state_5_null"]["G"]) <= 0.01 for x in r)
    c3 = sum(x["state_5_internal"]["G"] >= 0.02 and x["state_5_internal"]["lo"] > 0 for x in r)
    m5 = float(np.median([x["state_5_internal"]["G"] for x in r]))
    m20 = float(np.median([x["state_20_internal"]["G"] for x in r]))
    oth = [x for x in r if "state_5_other" in x]
    c5 = sum(x["state_5_other"]["lo"] > 0 for x in oth)
    e62 = res.get("e62", {})
    pairs = [(e62[f"{x['session']}|{x['area']}"], x["rate_5_internal"]["G"]) for x in r
             if f"{x['session']}|{x['area']}" in e62]
    rho = _spearman(*zip(*pairs)) if len(pairs) > 2 else float("nan")
    pre = [x["state_5_internal"]["periods"].get("pre", {}).get("G") for x in r]
    resp = [x["state_5_internal"]["periods"].get("response", {}).get("G") for x in r]
    mp = float(np.median([v for v in pre if v is not None]))
    mr = float(np.median([v for v in resp if v is not None]))
    return [
        ("C1", "calibration: planted signal raises G by >= 0.025 (lower > 0) in >= 90% of areas", c1 >= 0.9 * m,
         f"{c1}/{m}"),
        ("C2", "null: |G_null(state)| <= 0.01 at 50 ms in >= 90% of areas", c2 >= 0.9 * m, f"{c2}/{m}"),
        ("C3", "open inside at 50 ms: G_internal(state) >= 0.02 (lower > 0) in >= 50%", c3 >= 0.5 * m, f"{c3}/{m}"),
        ("C4", "closure grows with horizon: median G_internal(state) 200 ms < 50 ms", m20 < m5,
         f"50 ms {m5:+.3f}, 200 ms {m20:+.3f}"),
        ("C5", "open to other areas: G_other(state) lower > 0 at 50 ms in >= 50%", c5 >= 0.5 * len(oth),
         f"{c5}/{len(oth)}"),
        ("C6", "interaction makes levels open: Spearman(e62 S_proj(E2), G_internal(rate) 50 ms) >= 0.3", rho >= 0.3,
         f"rho {rho:+.2f} over {len(pairs)} areas"),
        ("C7", "the stimulus opens the level: median G_internal(state) response > pre-stimulus", mr > mp,
         f"pre {mp:+.3f}, response {mr:+.3f}"),
    ]


def analyze(res) -> str:
    lines = ["Experiment 63: is a brain area's population state a closed level?", "",
             "  G (state, top 3 components) at 50 / 100 / 200 ms: internal | other areas | null; planted gain"]
    for x in sorted(res["real"], key=lambda x: (x["area"], x["session"])):
        g = lambda k: x[k]["G"] if k in x else float("nan")  # noqa: E731
        pl = x["planted"]["after"]["G"] - x["planted"]["before"]["G"]
        lines.append(f"  {x['area']:6} s{x['session']:02d} n {x['n']:3d}  int " +
                     " ".join(f"{g(f'state_{h}_internal'):+.3f}" for h in HORIZONS) +
                     "  oth " + " ".join(f"{g(f'state_{h}_other'):+.3f}" for h in HORIZONS) +
                     f"  null {g('state_5_null'):+.3f}  rate-int {g('rate_5_internal'):+.3f}  planted +{pl:.3f}")
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
    sessions = [0] if quick else list(range(39))
    with ProcessPoolExecutor(max_workers=6) as pool:
        real = [r for rs in pool.map(_session_run, sessions) for r in rs]
    res = {"real": real, "e62": {f"{s}|{a}": v for (s, a), v in _e62(out_dir).items()}}
    (out_dir / "e63_area_closure_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e63_figures import make_figures
    make_figures(res, out_dir / "e63_figures", text)
    return text


if __name__ == "__main__":
    main()
