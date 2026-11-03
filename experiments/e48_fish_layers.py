"""Experiment 48: does schooling create a closed group-level layer? Real fish.

e47 showed, in simulations, that the spectrum's most interaction-blind block
can be a closed layer whose slow dynamics are made by interaction. Here the
same questions are asked of the six 8-fish recordings of Puy et al. (2024),
with their virtual groups (each fish time-shifted so they never met) as the
control. Only 8- and 39-fish groups exist in the dataset, so the group-size
test (layer size = group size minus one) cannot be done; this is the test
that can.

State dictionary (40 observables per 0.1 s step): every fish's x, y,
cos(heading), sin(heading) and speed.
Group block (5 variables): the school's centroid (x, y), mean heading vector
(mean cos, mean sin) and mean speed. Single-fish block: fish 0's 5 variables.

Closure, from data alone. For a block y = W' psi and horizon h,
    G(h) = 1 - tr(V^-1 MSE_parts) / tr(V^-1 MSE_block),  V = Cov(y_{t+h} - y_t),
MSE_block: least-squares forecast of y_{t+h} - y_t from the block's own past
  (y_t ... y_{t-4}, plus the quadratic terms of y_t);
MSE_parts: the same features plus every fish's full state at t, t-1, t-2
  (nested, so G > 0 means the parts carry information the block lacks).
Fitted on the first 70% of each recording, measured on the last 30%.

Interaction share at horizon h, S_h: e44's model with rule k6-AC (per-fish
turning and speed maps, shared social terms), fitted on the first 70%, run
forward from the observed state as a 32-member ensemble with and without its
social terms (the estimated twin, as in e46); full vs twin forecast error of
the block, whitened as above, with the ensemble's sampling error removed.
Spectrum: the generalized eigenproblem of the two 1 s (h = 10) forecast error
covariances over the 40 observables, weights from the first half of the test
starts, every S measured on the second half.

Predictions, fixed before running:
  F1  Real schools: the 5 most interaction-blind discovered variables (1 s
      spectrum) span the group block: smallest canonical cosine >= 0.7 in
      >= 5 of 6.
  F2  Real schools: the group block is nearly closed, G <= 0.1 at h = 1 s
      and 3 s, in >= 5 of 6.
  F3  Control: the group block is more closed in the real school than in
      its virtual pair, at both horizons, in >= 5 of 6.
  F4  Emergent: S_h(group block, 3 s) >= 0.05 in >= 5 of 6 real schools,
      and <= 0.02 in >= 5 of 6 virtual groups.
  F5  A single fish is not a layer: G(fish 0, 1 s) > G(group block, 1 s) in
      >= 5 of 6 real schools.
Exploratory: the full 40-variable spectrum (where is the gap?), G and S_h of
the discovered bottom-5 and top-5 blocks and of random 5-dim blocks.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEED = 48
RULE = "k6-AC"
H_SPEC = 10
HORIZONS = (10, 30)
LAGS, PART_LAGS = 5, 3
ENS, STARTS, N_RANDOM = 32, 1200, 10
N_FISH = 8
NAMES = ("x", "y", "cos", "sin", "speed")


def dictionary(p, th, sp):
    return np.concatenate([p[..., 0], p[..., 1], np.cos(th), np.sin(th), sp], -1)


def group_block(n):
    w = np.zeros((5 * n, 5))
    for b in range(5):
        w[b * n:(b + 1) * n, b] = 1.0 / n
    return w


def fish_block(n, i):
    w = np.zeros((5 * n, 5))
    for b in range(5):
        w[b * n + i, b] = 1.0
    return w


def overlap(w, target):
    qa, _ = np.linalg.qr(w)
    qb, _ = np.linalg.qr(target)
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False).min())


def ratio_gap(v, p_a, p_b):
    vi = np.linalg.inv(v)
    return float(1 - np.trace(vi @ p_a) / np.trace(vi @ p_b))


def gen_spectrum_cov(sig, sig0):
    li = np.linalg.inv(np.linalg.cholesky(sig0))
    lam, v = np.linalg.eigh(li @ sig @ li.T)
    w = li.T @ v
    return w / np.linalg.norm(w, axis=0), 1 - lam


def _lstsq(x_tr, y_tr, x_te):
    mu, sd = x_tr.mean(0), x_tr.std(0)
    sd[sd == 0] = 1.0
    mu[0], sd[0] = 0.0, 1.0                     # intercept column
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    coef = np.linalg.solve(a.T @ a + 1e-6 * len(a) * np.eye(a.shape[1]), a.T @ y_tr)
    return b @ coef


def closure_data(psi, wb, train, test, h):
    y = psi @ wb
    iu = np.triu_indices(wb.shape[1])

    def block_f(rows):
        cols = [np.ones((len(rows), 1))] + [y[rows - j] for j in range(LAGS)]
        cols.append((y[rows][:, :, None] * y[rows][:, None, :])[:, iu[0], iu[1]])
        return np.concatenate(cols, 1)

    def parts_f(rows):
        return np.concatenate([block_f(rows)] + [psi[rows - j] for j in range(PART_LAGS)], 1)

    d_tr, d_te = y[train + h] - y[train], y[test + h] - y[test]
    e_block = d_te - _lstsq(block_f(train), d_tr, block_f(test))
    e_parts = d_te - _lstsq(parts_f(train), d_tr, parts_f(test))
    v = np.cov(d_te.T)
    return ratio_gap(v, e_parts.T @ e_parts / len(e_parts), e_block.T @ e_block / len(e_block))


def _run(args):
    rec, kind = args
    import experiments.e44_elimination_speed_heading as m
    from experiments.e31_real_fish import kinematics, virtual
    from experiments.fish_data import load
    pos = load(N_FISH, rec)
    p, th, sp = kinematics(pos if kind == "real" else virtual(pos))
    seam = (sp > 100.0).any(1)
    sp = np.where(sp > 100.0, 0.0, sp)
    T = len(th)
    cut = int(0.7 * T)
    fits, _, _, _, (l1, l2, _, dvp) = m.fit_all(p, th, sp, cut, [RULE])
    fit = fits[RULE]
    psi = dictionary(p, th, sp)
    n = N_FISH
    hmax = max(HORIZONS)

    bad = np.zeros(T, bool)
    for s in np.nonzero(seam)[0]:
        bad[max(0, s - hmax - 1):s + LAGS + 2] = True
    rows = np.arange(LAGS, T - hmax)
    rows = rows[~bad[rows]]
    train, test = rows[rows + hmax < cut], rows[rows >= cut]

    blocks = {"group": group_block(n), "fish0": fish_block(n, 0)}
    rng = np.random.default_rng(SEED + rec + (100 if kind == "virtual" else 0))
    starts = np.sort(rng.choice(test, min(STARTS, len(test)), replace=False))
    ens = {}
    for name, gain in (("full", 1.0), ("twin", 0.0)):
        r = np.random.default_rng(SEED + 7 + rec)
        st = tuple(np.repeat(a[starts][:, None], ENS, 1) for a in (p, th, np.clip(sp, 0, m.base.V_MAX), l1, l2, dvp))
        out = {}
        for step in range(1, hmax + 1):
            st = m.model_step(st, fit, RULE, r, gain)
            if step in HORIZONS or step == H_SPEC:
                q = dictionary(st[0], st[1], st[2])
                mu = q.mean(1)
                dev = q - mu[:, None, :]
                out[step] = (mu, np.einsum("sri,srj->sij", dev, dev) / (ENS - 1))
        ens[name] = out

    def err_cov(name, h, sel):
        mu, cov = ens[name][h]
        e = psi[starts[sel] + h] - mu[sel]
        return e.T @ e / len(e) - cov[sel].mean(0) / ENS

    half = len(starts) // 2
    first, second = slice(0, half), slice(half, None)
    sig, sig0 = err_cov("full", H_SPEC, first), err_cov("twin", H_SPEC, first)
    w, _ = gen_spectrum_cov(sig, sig0)
    s_full, s_twin = err_cov("full", H_SPEC, second), err_cov("twin", H_SPEC, second)
    s_test = 1 - np.einsum("ik,ij,jk->k", w, s_full, w) / np.einsum("ik,ij,jk->k", w, s_twin, w)
    blocks["bottom5"], blocks["top5"] = w[:, -5:], w[:, :5]
    rr = np.random.default_rng(SEED + 50)
    for k in range(N_RANDOM):
        blocks[f"random{k}"] = np.linalg.qr(rr.standard_normal((5 * n, 5)))[0]

    def share_h(wb, h):
        d = psi[starts[second] + h] @ wb - psi[starts[second]] @ wb
        v = np.cov(d.T)
        return ratio_gap(v, wb.T @ err_cov("full", h, second) @ wb, wb.T @ err_cov("twin", h, second) @ wb)

    res = {"rec": rec, "kind": kind, "T": T, "train_rows": len(train), "test_rows": len(test),
           "S_test": s_test.tolist(), "w": w.tolist(),
           "overlap_group": overlap(w[:, -5:], blocks["group"]),
           "G": {b: [closure_data(psi, wb, train, test, h) for h in HORIZONS] for b, wb in blocks.items()},
           "Sh": {b: [share_h(wb, h) for h in HORIZONS] for b, wb in blocks.items()},
           "social_turn": fit[1].tolist(), "social_speed": fit[4].tolist()}
    return res


def _pairs(res):
    real = sorted([r for r in res["fish"] if r["kind"] == "real"], key=lambda r: r["rec"])
    virt = sorted([r for r in res["fish"] if r["kind"] == "virtual"], key=lambda r: r["rec"])
    return real, virt


def predictions(res):
    real, virt = _pairs(res)
    f1 = sum(r["overlap_group"] >= 0.7 for r in real)
    f2 = sum(max(r["G"]["group"]) <= 0.1 for r in real)
    f3 = sum(all(a < b for a, b in zip(r["G"]["group"], v["G"]["group"])) for r, v in zip(real, virt))
    f4a = sum(r["Sh"]["group"][1] >= 0.05 for r in real)
    f4b = sum(v["Sh"]["group"][1] <= 0.02 for v in virt)
    f5 = sum(r["G"]["fish0"][0] > r["G"]["group"][0] for r in real)
    return [
        ("F1", "real: 5 most blind variables span the group block (overlap >= 0.7, >= 5/6)", f1 >= 5,
         "overlap " + ", ".join(f"{r['overlap_group']:.2f}" for r in real)),
        ("F2", "real: group block nearly closed (G <= 0.1 at 1 s and 3 s, >= 5/6)", f2 >= 5,
         "G(1 s, 3 s) " + ", ".join(f"{r['G']['group'][0]:.3f}/{r['G']['group'][1]:.3f}" for r in real)),
        ("F3", "control: group block more closed in real than virtual (both h, >= 5/6)", f3 >= 5,
         "virtual G " + ", ".join(f"{v['G']['group'][0]:.3f}/{v['G']['group'][1]:.3f}" for v in virt)),
        ("F4", "emergent: S_3s(group) >= 0.05 real (>= 5/6), <= 0.02 virtual (>= 5/6)", f4a >= 5 and f4b >= 5,
         "real " + ", ".join(f"{r['Sh']['group'][1]:.3f}" for r in real)
         + "; virtual " + ", ".join(f"{v['Sh']['group'][1]:.3f}" for v in virt)),
        ("F5", "real: single fish less closed than the group (1 s, >= 5/6)", f5 >= 5,
         "G(fish 0) " + ", ".join(f"{r['G']['fish0'][0]:.3f}" for r in real)),
    ]


def analyze(res) -> str:
    real, virt = _pairs(res)
    lines = ["Experiment 48: does schooling create a closed group-level layer? (8-fish recordings)", "",
             "  G = closure gap (0 = closed), S_h = interaction share; each at 1 s / 3 s"]
    for r in real + virt:
        rnd = np.median([r["G"][f"random{k}"] for k in range(N_RANDOM)], 0)
        lines.append(f"  {r['kind']:7} #{r['rec']}  group G {r['G']['group'][0]:+.3f}/{r['G']['group'][1]:+.3f}"
                     f" S_h {r['Sh']['group'][0]:+.3f}/{r['Sh']['group'][1]:+.3f}"
                     f" | fish0 G {r['G']['fish0'][0]:+.3f}/{r['G']['fish0'][1]:+.3f}"
                     f" | bottom5 G {r['G']['bottom5'][0]:+.3f}/{r['G']['bottom5'][1]:+.3f}"
                     f" S_h {r['Sh']['bottom5'][1]:+.3f} overlap {r['overlap_group']:.2f}"
                     f" | top5 G {r['G']['top5'][0]:+.3f} | random G {rnd[0]:+.3f}/{rnd[1]:+.3f}")
    lines.append("")
    for r in real:
        lines.append(f"  real #{r['rec']} spectrum (most blind first): "
                     + " ".join(f"{x:.3f}" for x in r["S_test"][::-1][:12]) + " ...")
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
    with ProcessPoolExecutor(max_workers=8) as pool:
        fish = list(pool.map(_run, [(r, k) for r in range(6) for k in ("real", "virtual")]))
    res = {"fish": fish, "horizons": list(HORIZONS)}
    (out_dir / "e48_fish_layers_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e48_figures import make_figures
    make_figures(res, out_dir / "e48_figures", text)
    return text


if __name__ == "__main__":
    main()
