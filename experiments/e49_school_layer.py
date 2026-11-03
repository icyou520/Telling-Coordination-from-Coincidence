"""Experiment 49: relative headings across group size, and the school's shape as
a layer. Real fish (Puy et al. 2024), 8- and 39-fish recordings.

e48 (exploratory) found that in every 8-fish school the interaction-made
variables of the 1 s spectrum are 14 = 2 x (8 - 1): each fish's heading
vector relative to the school's mean heading. e48 also failed to find a
group-level layer, for two diagnosed reasons: school *shape* (polarization,
spread) is nonlinear and was not in the dictionary, and the closure test's
all-fish forecaster was linear, so it could not see an individual fish's
nonlinear own behaviour (virtual groups looked as closed as real schools).

Part A, group size. Dictionary: every fish's (cos heading, sin heading), 2N
variables. Spectrum at 1 s (h = 10 steps) from e44's model with rule k6-AC
(fitted on the first 70%), run forward from observed states with and without
its social terms (estimated twin, as in e46/e48); weights from half of the
test starts, S held out on the other half. Top block = everything above the
largest jump of the held-out spectrum sorted from most blind.
  8 fish: 1200 starts x 32 members.  39 fish: 2400 starts x 16 members.

Part B, the school's state as a layer (8 fish, real vs virtual). School state
Z (8 variables): centroid (x, y), mean heading vector (mean cos, mean sin),
mean speed, polarization |mean exp(i theta)|, spread (mean distance to the
centroid), milling |mean of r_hat x e_hat| (r_hat: unit vector from the
centroid, e_hat: heading).
  Interaction share per shape variable, S_h = 1 - MSE_full / MSE_twin, from
  the same ensembles (each member's polarization / spread, then the mean).
  Closure, from data alone: forecasting Z_{t+h} - Z_t from
    block features: Z_t ... Z_{t-4}, quadratic terms of Z_t, and the school's
      own place-and-heading map (25 place bumps at the centroid x (1, sin, cos)
      of the mean heading angle), against
    block + parts features: the above plus the same map averaged over the
      fish at their own positions and headings, and the mean velocity vector.
  If the fish move as a unit, their individual maps add nothing (G ~ 0); for
  independent fish (virtual groups) the school's future is the average of the
  fish's own futures, which the school's state cannot supply (G > 0).
  G(h) = 1 - tr(V^-1 MSE_parts) / tr(V^-1 MSE_block), V = Cov(Z_{t+h} - Z_t);
  fitted on the first 70%, measured on the last 30%.
  Single fish (e48's linear test, now for all 8 fish).

Predictions, fixed before running:
  P1  8 fish, heading spectrum: top block = 14 and the 2 most blind directions
      span the mean heading vector (smallest canonical cosine >= 0.9), in
      >= 5 of 6 real schools.
  P2  39 fish: top block = 76 = 2 x 38 and the 2 most blind directions span
      the mean heading vector (>= 0.9), in both recordings.
  P3  Shape is interaction-made: S_h(polarization) and S_h(spread) >= 0.3 at
      1 s and 3 s in >= 5 of 6 real schools; |S_h| <= 0.05 for both, both
      horizons, in >= 5 of 6 virtual groups.
  P4  The school state is closed against the fish's own tendencies:
      G <= 0.05 at 1 s and 3 s in >= 5 of 6 real schools.
  P5  Control: G(real) < G(virtual pair) at both horizons in >= 5 of 6.
  P6  Individual fish are not closed in schools: median over the 8 fish of
      G(fish i, 1 s) >= 0.05 in >= 5 of 6 real schools and <= 0.02 in >= 5 of
      6 virtual groups.
Exploratory: the 39-fish spectrum's shape; S_h of milling; S_h of shape in 39
fish (1 s).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.e48_fish_layers import closure_data, fish_block, gen_spectrum_cov, overlap, ratio_gap, _lstsq

SEED = 49
RULE = "k6-AC"
H_SPEC = 10
HORIZONS = (10, 30)
LAGS = 5
SETTINGS = {8: (1200, 32, (10, 30)), 39: (2400, 16, (10,))}   # starts, members, horizons
CHUNK = 300
Z_NAMES = ("cx", "cy", "mean cos", "mean sin", "mean speed", "polarization", "spread", "milling")


def headings(th):
    return np.concatenate([np.cos(th), np.sin(th)], -1)


def school_state(p, th, sp):
    c = p.mean(-2)
    mc, ms = np.cos(th).mean(-1), np.sin(th).mean(-1)
    r = p - c[..., None, :]
    dist = np.linalg.norm(r, axis=-1)
    rh = r / np.maximum(dist, 1e-9)[..., None]
    mill = np.abs((rh[..., 0] * np.sin(th) - rh[..., 1] * np.cos(th)).mean(-1))
    return np.stack([c[..., 0], c[..., 1], mc, ms, sp.mean(-1), np.hypot(mc, ms), dist.mean(-1), mill], -1)


def mean_heading_block(n):
    w = np.zeros((2 * n, 2))
    w[:n, 0] = w[n:, 1] = 1.0 / n
    return w


def top_block_size(s_test):
    asc = np.asarray(s_test)[::-1]
    return len(asc) - (int(np.argmax(np.diff(asc))) + 1)


def ensembles(m, state, starts, fit, gain, ens, hset, seed):
    """Per start: ensemble mean and covariance of the heading dictionary, mean of school state."""
    hmax = max(hset)
    out = {h: {"mu": [], "cov": [], "z": []} for h in hset}
    r = np.random.default_rng(seed)
    for a in range(0, len(starts), CHUNK):
        idx = starts[a:a + CHUNK]
        st = tuple(np.repeat(x[idx][:, None], ens, 1) for x in state)
        for step in range(1, hmax + 1):
            st = m.model_step(st, fit, RULE, r, gain)
            if step in hset:
                q = headings(st[1])
                mu = q.mean(1)
                dev = q - mu[:, None, :]
                out[step]["mu"].append(mu)
                out[step]["cov"].append(np.einsum("sri,srj->sij", dev, dev) / (ens - 1))
                out[step]["z"].append(school_state(st[0], st[1], st[2]).mean(1))
    return {h: {k: np.concatenate(v) for k, v in d.items()} for h, d in out.items()}


def heading_spectrum(q, starts, full, twin, ens, n):
    half = len(starts) // 2

    def err_cov(e, sel):
        x = q[starts[sel] + H_SPEC] - e["mu"][sel]
        return x.T @ x / len(x) - e["cov"][sel].mean(0) / ens

    first, second = slice(0, half), slice(half, None)
    w, _ = gen_spectrum_cov(err_cov(full, first), err_cov(twin, first))
    sf, st = err_cov(full, second), err_cov(twin, second)
    s_test = 1 - np.einsum("ik,ij,jk->k", w, sf, w) / np.einsum("ik,ij,jk->k", w, st, w)
    top = top_block_size(s_test)
    u = mean_heading_block(n) * np.sqrt(n)
    wt = w[:, :top] / np.linalg.norm(w[:, :top], axis=0)
    rel = 1 - ((u.T @ wt) ** 2).sum() / (wt ** 2).sum()
    return {"S_test": s_test.tolist(), "top": top, "overlap_mean_heading": overlap(w[:, -2:], u),
            "relative_fraction": float(rel)}


def school_closure(p, th, sp, train, test, h):
    import experiments.e43_elimination_fish_maps as base
    z = school_state(p, th, sp)
    ang = np.arctan2(z[:, 3], z[:, 2])
    b_c = base.bumps(z[:, :2])
    own_school = np.concatenate([b_c, b_c * np.sin(ang)[:, None], b_c * np.cos(ang)[:, None]], 1)
    b_f = base.bumps(p)
    own_fish = np.concatenate([b_f.mean(1), (b_f * np.sin(th)[..., None]).mean(1),
                               (b_f * np.cos(th)[..., None]).mean(1),
                               (sp * np.cos(th)).mean(1)[:, None], (sp * np.sin(th)).mean(1)[:, None]], 1)
    iu = np.triu_indices(z.shape[1])

    def block_f(rows):
        cols = [np.ones((len(rows), 1))] + [z[rows - j] for j in range(LAGS)]
        cols += [(z[rows][:, :, None] * z[rows][:, None, :])[:, iu[0], iu[1]], own_school[rows]]
        return np.concatenate(cols, 1)

    def parts_f(rows):
        return np.concatenate([block_f(rows), own_fish[rows]], 1)

    d_tr, d_te = z[train + h] - z[train], z[test + h] - z[test]
    e_b = d_te - _lstsq(block_f(train), d_tr, block_f(test))
    e_p = d_te - _lstsq(parts_f(train), d_tr, parts_f(test))
    return ratio_gap(np.cov(d_te.T), e_p.T @ e_p / len(e_p), e_b.T @ e_b / len(e_b))


def _run(args):
    n, rec, kind = args
    import experiments.e44_elimination_speed_heading as m
    from experiments.e31_real_fish import kinematics, virtual
    from experiments.fish_data import load
    pos = load(n, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    seam = (sp > 100.0).any(1)
    sp = np.where(sp > 100.0, 0.0, sp)
    T = len(th)
    cut = int(0.7 * T)
    n_starts, ens, hset = SETTINGS[n]
    hmax = max(hset)
    fits, _, _, _, (l1, l2, _, dvp) = m.fit_all(p, th, sp, cut, [RULE])
    fit = fits[RULE]
    bad = np.zeros(T, bool)
    for s in np.nonzero(seam)[0]:
        bad[max(0, s - max(HORIZONS) - 1):s + LAGS + 2] = True
    rows = np.arange(LAGS, T - max(HORIZONS))
    rows = rows[~bad[rows]]
    train, test = rows[rows + max(HORIZONS) < cut], rows[rows >= cut]
    rng = np.random.default_rng(SEED + 10 * n + rec + (5 if kind == "virtual" else 0))
    starts = np.sort(rng.choice(test[test < T - hmax - 1], min(n_starts, len(test)), replace=False))
    state = (p, th, np.clip(sp, 0, m.base.V_MAX), l1, l2, dvp)
    full = ensembles(m, state, starts, fit, 1.0, ens, hset, SEED + 7 + rec)
    twin = ensembles(m, state, starts, fit, 0.0, ens, hset, SEED + 7 + rec)
    q = headings(th)
    res = {"N": n, "rec": rec, "kind": kind, "spectrum": heading_spectrum(q, starts, full[H_SPEC], twin[H_SPEC], ens, n)}
    z = school_state(p, th, sp)
    sh = {}
    for h in hset:
        truth = z[starts + h]
        ef, et = ((truth - full[h]["z"]) ** 2).mean(0), ((truth - twin[h]["z"]) ** 2).mean(0)
        sh[h] = (1 - ef / et).tolist()
        res[f"obs_vs_forecast_{h}"] = {"obs": truth.mean(0).tolist(), "full": full[h]["z"].mean(0).tolist(),
                                       "twin": twin[h]["z"].mean(0).tolist()}
    res["Sh"] = {str(h): v for h, v in sh.items()}
    if n == 8:
        psi = np.concatenate([p[..., 0], p[..., 1], np.cos(th), np.sin(th), sp], -1)
        res["G_school"] = [school_closure(p, th, sp, train, test, h) for h in HORIZONS]
        res["G_fish"] = [closure_data(psi, fish_block(n, i), train, test, HORIZONS[0]) for i in range(n)]
    return res


def _sel(res, n, kind):
    return sorted([r for r in res["runs"] if r["N"] == n and r["kind"] == kind], key=lambda r: r["rec"])


def predictions(res):
    real, virt, big = _sel(res, 8, "real"), _sel(res, 8, "virtual"), _sel(res, 39, "real")
    p1 = sum(r["spectrum"]["top"] == 14 and r["spectrum"]["overlap_mean_heading"] >= 0.9 for r in real)
    p2 = sum(r["spectrum"]["top"] == 76 and r["spectrum"]["overlap_mean_heading"] >= 0.9 for r in big)
    ip, isp = Z_NAMES.index("polarization"), Z_NAMES.index("spread")
    p3a = sum(all(r["Sh"][h][i] >= 0.3 for h in ("10", "30") for i in (ip, isp)) for r in real)
    p3b = sum(all(abs(v["Sh"][h][i]) <= 0.05 for h in ("10", "30") for i in (ip, isp)) for v in virt)
    p4 = sum(max(r["G_school"]) <= 0.05 for r in real)
    p5 = sum(all(a < b for a, b in zip(r["G_school"], v["G_school"])) for r, v in zip(real, virt))
    p6a = sum(np.median(r["G_fish"]) >= 0.05 for r in real)
    p6b = sum(np.median(v["G_fish"]) <= 0.02 for v in virt)
    return [
        ("P1", "8 fish: top block = 14, 2 most blind = mean heading (>= 5/6)", p1 >= 5,
         "top " + ", ".join(str(r["spectrum"]["top"]) for r in real) + "; overlap "
         + ", ".join(f"{r['spectrum']['overlap_mean_heading']:.2f}" for r in real)),
        ("P2", "39 fish: top block = 76, 2 most blind = mean heading (2/2)", p2 == 2,
         "top " + ", ".join(str(r["spectrum"]["top"]) for r in big) + "; overlap "
         + ", ".join(f"{r['spectrum']['overlap_mean_heading']:.2f}" for r in big)),
        ("P3", "shape interaction-made: S_h(pol, spread) >= 0.3 real, |.| <= 0.05 virtual", p3a >= 5 and p3b >= 5,
         "real pol/spread 3 s " + ", ".join(f"{r['Sh']['30'][ip]:.2f}/{r['Sh']['30'][isp]:.2f}" for r in real)
         + f"; virtual ok {p3b}/6"),
        ("P4", "school state closed vs fish's own tendencies (G <= 0.05, >= 5/6)", p4 >= 5,
         "G(1 s, 3 s) " + ", ".join(f"{r['G_school'][0]:.3f}/{r['G_school'][1]:.3f}" for r in real)),
        ("P5", "control: G(real) < G(virtual), both h (>= 5/6)", p5 >= 5,
         "virtual " + ", ".join(f"{v['G_school'][0]:.3f}/{v['G_school'][1]:.3f}" for v in virt)),
        ("P6", "single fish: median G >= 0.05 real, <= 0.02 virtual (>= 5/6 each)", p6a >= 5 and p6b >= 5,
         "real " + ", ".join(f"{np.median(r['G_fish']):.3f}" for r in real)
         + "; virtual " + ", ".join(f"{np.median(v['G_fish']):.3f}" for v in virt)),
    ]


def analyze(res) -> str:
    lines = ["Experiment 49: relative headings across group size; the school's shape as a layer", ""]
    for r in sorted(res["runs"], key=lambda r: (r["N"], r["kind"], r["rec"])):
        s = r["spectrum"]
        line = (f"  {r['N']:2d} fish {r['kind']:7} #{r['rec']}: top block {s['top']:3d} "
                f"(relative {s['relative_fraction']:.3f}), blind-2 vs mean heading {s['overlap_mean_heading']:.2f}; "
                "S_h pol/spread/mill " + " | ".join(
                    f"{int(h) / 10:.0f} s {v[5]:+.2f}/{v[6]:+.2f}/{v[7]:+.2f}" for h, v in r["Sh"].items()))
        if "G_school" in r:
            line += (f"; G school {r['G_school'][0]:+.3f}/{r['G_school'][1]:+.3f}"
                     f"; G fish median {np.median(r['G_fish']):+.3f}")
        lines.append(line)
    for r in _sel(res, 39, "real"):
        asc = np.array(r["spectrum"]["S_test"])[::-1]
        lines.append(f"  39 fish #{r['rec']} spectrum, most blind first: "
                     + " ".join(f"{x:.2f}" for x in asc[:6]) + " ... "
                     + " ".join(f"{x:.2f}" for x in asc[-4:]))
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
    (out_dir / "e49_school_layer_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e49_figures import make_figures
    make_figures(res, out_dir / "e49_figures", text)
    return text


if __name__ == "__main__":
    main()
