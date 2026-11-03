"""Experiment 56: does a group's openness follow how loosely its members are measured to be locked to it?

e55 assigned freedom by design. In the oscillators one free member out of 30 opened the group, with the
information in that member; in the flock, privately informed members changed little and the information sat
in the others. e56 stops assigning freedom and measures it. Each member's looseness is

    lambda_i = 1 - | mean_t exp(i phi_i(t)) |,   phi_i = its heading (or phase) relative to the group's,

over the fitted (first 70%) part of the run only: 0 = always aligned with the group, 1 = no relation.

Data:
  A  24 new oscillator groups (30 members), parameters drawn at random: coupling K in [1.0, 2.5], spread of
     natural frequencies in [0.1, 0.7], noise in [0.2, 0.6] (e55_sims.kuramoto_free with m = 0). Who is
     loose emerges from the draw.
  B  24 new flocks (30 agents), drawn at random: noise in [0.2, 0.6], alignment in [0.8, 2.0], speed spread
     in [0.05, 0.3], privately informed agents m in {0..4} (e55_sims.flock_free).
  C  real fish: the six 8-fish Puy et al. schools (0.1 s) and the nine zebrafish groups of 60-100 (e53).
Nothing about the drawn groups or the fish's looseness was looked at before the predictions were written.

Test: e54/e55's (calibrated closure gap G at one time unit, e51's ridge and bootstrap). Simulations: e54's
block and arrangement features; fish: e53's block (with a place map) and e52's influence features. Targets
as in e55: r and the phase drift (oscillators), the turn (flocks, fish). Two extra models use the
arrangement of only the k loosest or only the k tightest members (k = 3 in simulations, max(2, N/4) in fish).
Null: the arrangement of another group (simulations: same parameters, another seed; fish: as e52/e53).

Predictions, fixed before running:
  S0  The test works in the simulations: planted rise >= 0.025 with lower bound > 0 in >= 19/24 of each
      ensemble, null |G| <= 0.01 in >= 19/24 of each.
  A1  Oscillators: the loosest member sets how open r is: Spearman(max lambda, G(r)) >= 0.6.
  A2  Oscillators: the phase drift opens with the number of loose members (lambda > 0.5):
      Spearman(count, G(drift)) >= 0.6.
  A3  Oscillators: in runs where r is open (G >= 0.03, lower > 0), G(loosest k) > G(tightest k) in >= 80%.
  B1  Flocks: Spearman(max lambda, G(turn)) >= 0.6.
  B2  Flocks: in runs where the turn is open, G(loosest k) > G(tightest k) in >= 80%.
  C1  Fish: G(turn | loosest k) > G(turn | tightest k) in >= 11 of the 15 real groups.
  C2  Fish: the null is clean, |G| <= 0.01 in >= 12 of 15.
"""

from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e49_school_layer import LAGS, school_state
from experiments.e50_school_configuration import _wrap
from experiments.e51_trusted_closure import PLANT, Ridge
from experiments.e52_influence_map import _bump, _tame_fish, bumps, fish_frame, null_recordings
from experiments.e53_influence_replication import ZEBRA, _rows, gap_ci, place_map
from experiments.e54_neutral_directions import _lags, _quad, _rates, _spearman
from experiments.e55_free_members import phase_sets
from experiments.e55_sims import flock_free, kuramoto_free

SEED = 56
RUNS = 24
STEPS = 25000
H = 10
K_SIM = 3
LOOSE = 0.5


def draw(system, i):
    rng = np.random.default_rng(SEED * 100 + i + (1000 if system == "flock" else 0))
    if system == "kuramoto":
        return {"coupling": float(rng.uniform(1.0, 2.5)), "locked_sd": float(rng.uniform(0.1, 0.7)),
                "noise": float(rng.uniform(0.2, 0.6))}
    return {"noise": float(rng.uniform(0.2, 0.6)), "align": float(rng.uniform(0.8, 2.0)),
            "speed_sd": float(rng.uniform(0.05, 0.3)), "m": int(rng.integers(0, 5))}


def looseness(th, ref):
    psi = np.angle(np.exp(1j * th).mean(1))
    return 1 - np.abs(np.exp(1j * (th[ref] - psi[ref, None])).mean(0))


def influence_sets(p, th, sp, ref, sets):
    """e52's 75 influence features, averaged over each index set; one feature at a time (memory)."""
    x, y, phi, omega, u = fish_frame(p, th, sp)
    om, ut = _tame_fish(omega, ref), _tame_fish(u, ref)
    ch = {"place": np.ones_like(x), "sin": np.sin(phi), "cos": np.cos(phi), "turn": om, "speed": ut,
          "speed_sin": ut * np.sin(phi)}
    cols = [[] for _ in sets]
    for c, par, a, b in bumps():
        f = ch[c] * (_bump(x, y, a, b) - par * _bump(x, -y, a, b))
        for j, idx in enumerate(sets):
            cols[j].append(f[:, idx].mean(1))
    return [np.column_stack(c) for c in cols]


def _sets(lam, k):
    order = np.argsort(lam)
    return [np.arange(len(lam)), order[-k:], order[:k]]


def _closure(block, extras, targets, train, test, gap, rng, plant_target, plant_feats):
    rows = np.concatenate([train, test])
    pos_tr, pos_te = np.arange(len(train)), np.arange(len(train), len(rows))
    nb = block.shape[1]
    models = {"block": Ridge(block, train, test, gap)}
    models.update({k: Ridge(np.concatenate([block, x], 1), train, test, gap, nb) for k, x in extras.items()})
    out = {}
    for name, y in targets.items():
        y_tr, y_te = y[pos_tr], y[pos_te]
        w = 1 / y_tr.var(0)
        vinv = np.linalg.inv(np.atleast_2d(np.cov(y_te.T)))
        q = {}
        for k, m in models.items():
            e = y_te - m.fit(y_tr, w)[0]
            q[k] = np.einsum("ti,ij,tj->t", e, vinv, e)
        out[name] = {k: gap_ci(q[k], q["block"], rng) for k in extras}
    if plant_feats is not None:
        y = targets[plant_target][:, 0]
        ones_block = np.column_stack([np.ones(len(block)), block])
        xi = plant_feats - ones_block @ np.linalg.lstsq(ones_block[rows], plant_feats[rows], rcond=None)[0]
        one = np.ones(1)
        pb = models["block"].fit(y[pos_tr, None], one)[0][:, 0]
        kappa = np.sqrt(PLANT / (1 - PLANT) * ((y[pos_te] - pb) ** 2).mean() / xi[test].var())
        yp = y + kappa * xi[rows]
        fb = models["block"].fit(yp[pos_tr, None], one)[0][:, 0]
        fa = models["all"].fit(yp[pos_tr, None], one)[0][:, 0]
        out["planted"] = {"before": out[plant_target]["all"],
                          "after": gap_ci((yp[pos_te] - fa) ** 2, (yp[pos_te] - fb) ** 2, rng)}
    return out


def _sim(system, i, seed, ref):
    prm = draw(system, i)
    if system == "kuramoto":
        th = kuramoto_free(m=0, steps=STEPS, seed=seed, **prm)
        lam = looseness(th, ref)
        feats = phase_sets(th, ref, _sets(lam, K_SIM))
        zc = np.exp(1j * th).mean(1)
        psi = np.angle(zc)
        z = np.column_stack([np.abs(zc), np.cos(psi), np.sin(psi)])
        stable = z[:, [0]]
    else:
        p, th, sp = flock_free(steps=STEPS, seed=seed, **prm)
        lam = looseness(th, ref)
        feats = influence_sets(p, th, sp, ref, _sets(lam, K_SIM))
        z = school_state(p, th, sp)[:, 2:]
        psi = np.arctan2(z[:, 1], z[:, 0])
        stable = z[:, [3, 4]]
    block = np.concatenate([_lags(z), _quad(z), _rates(psi)], 1)
    return prm, lam, block, feats, psi, stable


def _run_sim(args):
    system, i = args
    rng = np.random.default_rng(SEED * 1000 + i + (500 if system == "flock" else 0))
    cut = int(0.7 * STEPS)
    ref = np.arange(cut)
    prm, lam, block, feats, psi, stable = _sim(system, i, 0, ref)
    null = _sim(system, i, 100, ref)[3][0]
    rows = np.arange(LAGS + 1, STEPS - H)
    train, test = rows[rows + H < cut], rows[rows >= cut]
    rows = np.concatenate([train, test])
    targets = {"turn": _wrap(psi[rows + H] - psi[rows])[:, None], "stable": stable[rows + H] - stable[rows]}
    extras = {"all": feats[0], "loosest": feats[1], "tightest": feats[2], "null": null}
    j0 = bumps().index(("sin", -1, 0.75, 0.0)) if system == "flock" else 2
    plant = "stable" if system == "kuramoto" else "turn"
    out = _closure(block, extras, targets, train, test, H + LAGS, rng, plant, feats[0][:, j0])
    return {"system": system, "i": i, "params": prm, "lam": lam.tolist(), "max_lam": float(lam.max()),
            "n_loose": int((lam > LOOSE).sum()), **out}


def _fish(spec):
    if spec[0] == "puy":
        from experiments.e31_real_fish import kinematics
        from experiments.fish_data import load
        p, th, sp = kinematics(load(8, spec[1]))
        sp = np.where(sp > 100.0, 0.0, sp)
        brk = np.zeros(len(th), bool)
        brk[0] = True
        return p, th, sp, brk, 0.1
    from experiments.e53_influence_replication import zebrafish
    return zebrafish(spec[1], spec[2], "real")


def _run_fish(args):
    spec, null_spec = args
    rng = np.random.default_rng(SEED * 1000 + zlib.crc32(repr(spec).encode()) % 997)
    p, th, sp, brk, dt = _fish(spec)
    T, n = th.shape
    ref = np.arange(int(0.7 * T))
    lam = looseness(th, ref)
    k = max(2, n // 4)
    feats = influence_sets(p, th, sp, ref, _sets(lam, k))
    po, tho, spo, _, _ = _fish(null_spec)
    null = np.resize(influence_sets(po, tho, spo, np.arange(len(tho)), [np.arange(tho.shape[1])])[0], feats[0].shape)
    z = school_state(p, th, sp)
    psi = np.arctan2(z[:, 3], z[:, 2])
    block = np.concatenate([_lags(z), _quad(z), place_map(z, ref), _rates(psi)], 1)
    h = int(round(1 / dt))
    train, test = _rows(brk, T, h)
    rows = np.concatenate([train, test])
    targets = {"turn": _wrap(psi[rows + h] - psi[rows])[:, None]}
    extras = {"all": feats[0], "loosest": feats[1], "tightest": feats[2], "null": null}
    out = _closure(block, extras, targets, train, test, h + LAGS, rng, "turn", None)
    return {"system": "fish", "group": f"{spec[0]} {'-'.join(map(str, spec[1:]))}", "N": n, "k": k,
            "lam": lam.tolist(), "max_lam": float(lam.max()), **out}


def _sel(res, system):
    return sorted([r for r in res["runs"] if r["system"] == system], key=lambda r: r.get("i", 0))


def predictions(res):
    out = []
    cal, nul = {}, {}
    for s in ("kuramoto", "flock"):
        rs = _sel(res, s)
        cal[s] = sum(r["planted"]["after"]["G"] - r["planted"]["before"]["G"] >= 0.025 and r["planted"]["after"]["lo"] > 0
                     for r in rs)
        nul[s] = sum(all(abs(v["null"]["G"]) <= 0.01 for t, v in r.items() if t in ("turn", "stable")) for r in rs)
    out.append(("S0", "the test works in the simulations (planted, null >= 19/24 in each)",
                min(cal.values()) >= 19 and min(nul.values()) >= 19,
                f"oscillators planted {cal['kuramoto']}/24 null {nul['kuramoto']}/24; "
                f"flocks planted {cal['flock']}/24 null {nul['flock']}/24"))

    def lt(rs, t):
        op = [r for r in rs if r[t]["all"]["G"] >= 0.03 and r[t]["all"]["lo"] > 0]
        win = sum(r[t]["loosest"]["G"] > r[t]["tightest"]["G"] for r in op)
        return op, win

    ks = _sel(res, "kuramoto")
    rho1 = _spearman([r["max_lam"] for r in ks], [r["stable"]["all"]["G"] for r in ks])
    rho2 = _spearman([r["n_loose"] for r in ks], [r["turn"]["all"]["G"] for r in ks])
    op, win = lt(ks, "stable")
    out += [("A1", "oscillators: Spearman(max looseness, G(r)) >= 0.6", rho1 >= 0.6, f"rho {rho1:+.2f}"),
            ("A2", "oscillators: Spearman(number of loose members, G(drift)) >= 0.6", rho2 >= 0.6,
             f"rho {rho2:+.2f}; counts {sorted(r['n_loose'] for r in ks)}"),
            ("A3", "oscillators: loosest k > tightest k where r is open (>= 80%)",
             bool(op) and win >= 0.8 * len(op), f"{win}/{len(op)}")]
    fs = _sel(res, "flock")
    rho = _spearman([r["max_lam"] for r in fs], [r["turn"]["all"]["G"] for r in fs])
    op, win = lt(fs, "turn")
    out += [("B1", "flocks: Spearman(max looseness, G(turn)) >= 0.6", rho >= 0.6, f"rho {rho:+.2f}"),
            ("B2", "flocks: loosest k > tightest k where the turn is open (>= 80%)",
             bool(op) and win >= 0.8 * len(op), f"{win}/{len(op)}")]
    fish = _sel(res, "fish")
    c1 = sum(r["turn"]["loosest"]["G"] > r["turn"]["tightest"]["G"] for r in fish)
    c2 = sum(abs(r["turn"]["null"]["G"]) <= 0.01 for r in fish)
    out += [("C1", f"fish: G(loosest) > G(tightest) in >= 11/{len(fish)}", c1 >= 11,
             f"{c1}/{len(fish)}; " + ", ".join(f"{r['turn']['loosest']['G']:+.3f}/{r['turn']['tightest']['G']:+.3f}"
                                               for r in fish)),
            ("C2", f"fish: null clean in >= 12/{len(fish)}", c2 >= 12, f"{c2}/{len(fish)}")]
    return out


def analyze(res) -> str:
    lines = ["Experiment 56: does openness follow measured looseness?", ""]
    for r in _sel(res, "kuramoto") + _sel(res, "flock") + _sel(res, "fish"):
        name = r.get("group") or f"{r['system']} {r['i']:2d}"
        ts = [t for t in ("turn", "stable") if t in r]
        g = "  ".join(f"{t}: all {r[t]['all']['G']:+.3f} [{r[t]['all']['lo']:+.3f}] loose {r[t]['loosest']['G']:+.3f}"
                      f" tight {r[t]['tightest']['G']:+.3f} null {r[t]['null']['G']:+.3f}" for t in ts)
        pl = f" | planted {r['planted']['before']['G']:+.3f} -> {r['planted']['after']['G']:+.3f}" if "planted" in r else ""
        lines.append(f"  {name:16} max lambda {r['max_lam']:.2f} loose {r.get('n_loose', '-')!s:>2} | {g}{pl}")
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def tasks():
    sims = [(s, i) for s in ("kuramoto", "flock") for i in range(RUNS)]
    nulls = null_recordings(8, 6)
    fish = [(("puy", r), ("puy", nulls[r])) for r in range(6)]
    fish += [(("zebrafish", n, r), ("zebrafish", *ZEBRA[(j + 3) % len(ZEBRA)])) for j, (n, r) in enumerate(ZEBRA)]
    return sims, fish


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    sims, fish = tasks()
    if quick:
        sims, fish = sims[:2] + sims[RUNS:RUNS + 2], fish[:1]
    with ProcessPoolExecutor(max_workers=7) as pool:
        runs = list(pool.map(_run_sim, sims)) + list(pool.map(_run_fish, fish))
    res = {"runs": runs}
    (out_dir / "e56_measured_lock_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e56_figures import make_figures
    make_figures(res, out_dir / "e56_figures", text)
    return text


if __name__ == "__main__":
    main()
