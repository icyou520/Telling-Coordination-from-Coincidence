"""Experiment 47: closed layers. Do the collective variables the spectrum
discovers form a self-contained level of description?

e45/e46 rank every collective variable by its one-step interaction share S.
A *layer* of organization asks for more: a small block of collective
variables whose future is predicted from the block's own past as well as from
the full micro state (closure), so the parts can be forgotten. A layer is
*emergent* when it is closed and its own slow dynamics are made by
interaction; it is merely *inherited* when it is closed because interactions
never touch it (the centre of mass).

Closure gap of a block y = W' psi (k columns) at horizon h, with
V = Cov(y_{t+h} - y_t), so G does not depend on how the block is mixed or
scaled:
    G(h) = 1 - tr(V^-1 MSE_micro) / tr(V^-1 MSE_macro)
MSE_micro: best forecast from the full micro state x_t (the true model's
conditional mean; exact for linear, a 64-member ensemble otherwise, with
the ensemble's sampling error removed). MSE_macro: best forecast from the
block's own past (least squares on y_t, ..., y_{t-9}; for the oscillators
also the quadratic terms of y_t). G = 0: closed.
Interaction share at horizon h, S_h: the same ratio with the full model's
forecast from x_t against the twin's forecast from x_t.

Theory (Lean, InteractionShare.lean Part 5). A left-invariant block
(W'A = M W') is closed (`left_invariant_closed`), on every horizon
(`closed_iter`), and a closed description of a closed description is a
closed description of the parts (`closed_comp`): layers stack. For an
interaction operator organised by scale, the left null space of the strong
part is the layer above it (Simon's near-decomposability).

Systems (simulated; true model and twin known):
  H  three-level hierarchy of N = 12 walkers: 2 super-modules x 2 modules x 3
     walkers; x' = (1 - theta) x - (L_in + L_mid + L_top) x + noise,
     theta = 0.005. Inside each module, one-way coupling on a random digraph
     (ring plus each other link with prob 0.5, weights log-uniform 0.3-3,
     scaled so the module's max out-degree is 0.9). Between the modules of a
     super-module, random directed links (prob 0.3) of weight 0.04; between
     super-modules, weight 0.006. True layers (from the model): module layer
     = left null space of L_in (4 pi-weighted module averages); super-module
     layer = left null space of L_in + L_mid (2); global = 1.
  K  modular Kuramoto: 3 modules x 4 oscillators, coupling K_in = 2 within
     and K_out = 0.2 between modules (each normalised by the number of
     partners), omega = module mean (-0.05, 0, 0.05) + 0.2 N(0,1), noise 0.3,
     dt = 0.05. Dictionary (cos theta_i, sin theta_i): 24 observables.
     Expected layer: the 3 module order parameters (Re, Im of the module's
     mean of exp(i theta)), 6 dims.
Each run (150 000 steps) is split in thirds: spectrum weights from the first,
the macro forecaster from the second, S, S_h and G measured on the last.

Automatic layer cuts (fixed rule): order directions from most blind upward
with held-out S; a layer of size k is declared where
S_(k+1) >= 4 max(S_(k), 0.0005).

Design facts for H (exact, from the model, before any simulation; design()).
Settings chosen at this stage: per-module weight scaling and weights 0.3-3
(the first design, one global scale with weights 0.1-10, left some modules
barely coupled inside, so the bottom-4 block was not the module layer), leak
theta 0.005 (at 0.1 the leak, not interaction, drove the slow layers), and
couplings 0.04 / 0.006 (from 0.03/0.003-0.008, for the largest gaps).
  exact S, most blind first: 0, 0.0044, 0.0374, 0.0593, then 0.335-0.69 (8);
      gap ratios 8.9, 8.4, (1.6), 5.7: cuts at k = 1, 2, 4.
  overlap of exact bottom-k blocks with the true layers: 1.0, 0.996, 0.994.
  G of the true module layer <= 0.003 at h = 1, 5, 20, 100; plain (unweighted)
      module averages G(1) = 0.245; most interaction-made 4 directions G(1) 0.47.
  S_h of the module layer at h = 20: 0.21; super-module layer at h = 100:
      0.077; global: 0.

Predictions, fixed before running:
  H1  Held-out spectrum within 0.02 of the exact one, every direction.
  H2  The automatic rule finds layers of exactly k = 1, 2 and 4.
  H3  Discovered layers = true layers: overlap (smallest canonical cosine)
      >= 0.98 for bottom-4 / module layer, bottom-2 / super-module layer,
      bottom-1 / global.
  H4  Discovered module and super-module layers are closed: G <= 0.02 at
      h = 1, 5, 20, 100.
  H5  Not every block is a layer: plain module averages G(1) >= 0.15, and
      the 4 most interaction-made directions G(1) >= 0.3.
  H6  Emergent, not inherited: S_h(module layer, h = 20) >= 0.1 and
      S_h(super-module layer, h = 100) >= 0.04; global |S_h| <= 0.01 at all h.
  K1  Bottom-6 discovered block = the module order parameters: overlap >= 0.9.
  K2  It is closed: G <= 0.05 at h = 1, 5, 20, 100.
  K3  It is more closed than the 6 most interaction-made directions and
      than the median of 20 random 6-dim blocks, at every h.
  K4  Emergent: one-step S of the block is small (<= 0.05) but
      S_h(h = 100) >= 0.05.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e45_interaction_spectrum import gen_spectrum, gen_spectrum_exact, share, stationary_cov

SEED = 47
THETA = 0.005
W_LO, W_HI, EPS_MID, EPS_TOP = 0.3, 3.0, 0.04, 0.006
T_STEPS, BURN, LAGS = 150_000, 2_000, 10
HORIZONS = (1, 5, 20, 100)
SIZES = (3, 2, 2)                      # walkers per module, modules per super, supers
RATIO, FLOOR = 4.0, 5e-4
STARTS, ENS, N_RANDOM = 800, 64, 20


# ---------------------------------------------------------------- shared
def thirds(n):
    a, b = n // 3, 2 * n // 3
    return slice(0, a), slice(a, b), slice(b, n)


def overlap(w, target):
    """Smallest canonical cosine between two subspaces (1 = identical)."""
    qa, _ = np.linalg.qr(w)
    qb, _ = np.linalg.qr(target)
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False).min())


def layer_cuts(s_test):
    """Layer sizes k from the held-out spectrum (s_test ordered most made -> most blind)."""
    asc = np.asarray(s_test)[::-1]
    return [k for k in range(1, len(asc)) if asc[k] >= RATIO * max(asc[k - 1], FLOOR)]


def ratio_gap(v, p_a, p_b):
    vi = np.linalg.inv(v)
    return float(1 - np.trace(vi @ p_a) / np.trace(vi @ p_b))


def macro_errors(y, rows_fit, rows_eval, h, quad=False):
    """Errors of the least-squares forecast of y_{t+h} - y_t from the block's own past."""
    def feats(rows):
        cols = [np.ones(len(rows))[:, None]] + [y[rows - j] for j in range(LAGS)]
        if quad:
            iu = np.triu_indices(y.shape[1])
            cols.append((y[rows][:, :, None] * y[rows][:, None, :])[:, iu[0], iu[1]])
        return np.concatenate(cols, 1)
    xf = feats(rows_fit)
    coef = np.linalg.solve(xf.T @ xf + 1e-8 * np.eye(xf.shape[1]), xf.T @ (y[rows_fit + h] - y[rows_fit]))
    return (y[rows_eval + h] - y[rows_eval]) - feats(rows_eval) @ coef


# ---------------------------------------------------------------- hierarchy
def lap(w):
    return np.diag(w.sum(1)) - w


def hierarchy(rng):
    m_size, n_mod, n_sup = SIZES
    n = m_size * n_mod * n_sup
    module = np.arange(n) // m_size
    sup = module // n_mod
    w_in = np.zeros((n, n))
    for m in range(n_mod * n_sup):
        idx = np.nonzero(module == m)[0]
        for a in range(m_size):
            i, j = idx[a], idx[(a + 1) % m_size]
            w_in[i, j] = np.exp(rng.uniform(np.log(W_LO), np.log(W_HI)))
        for i in idx:
            for j in idx:
                if i != j and w_in[i, j] == 0 and rng.random() < 0.5:
                    w_in[i, j] = np.exp(rng.uniform(np.log(W_LO), np.log(W_HI)))
        w_in[idx] *= 0.9 / w_in[idx].sum(1).max()
    same_sup = sup[:, None] == sup[None, :]
    diff_mod = module[:, None] != module[None, :]
    links = rng.random((n, n)) < 0.3
    w_mid = EPS_MID * (links & same_sup & diff_mod)
    w_top = EPS_TOP * (links & ~same_sup)
    l_in, l_mid, l_top = lap(w_in), lap(w_mid), lap(w_top)
    a0 = (1 - THETA) * np.eye(n)
    a = a0 - l_in - l_mid - l_top
    return {"A": a, "A0": a0, "L_in": l_in, "L_mid": l_mid, "L_top": l_top, "module": module, "sup": sup}


def left_null(m, k):
    """Basis of {w : w' m = 0} (the k smallest left singular vectors)."""
    u, _, _ = np.linalg.svd(m)
    return u[:, -k:]


def targets(h):
    n_mod, n_sup = h["module"].max() + 1, h["sup"].max() + 1
    plain_mod = np.stack([(h["module"] == m) / np.sqrt((h["module"] == m).sum()) for m in range(n_mod)], 1)
    return {"module": left_null(h["L_in"], n_mod), "super": left_null(h["L_in"] + h["L_mid"], n_sup),
            "global": left_null(h["L_in"] + h["L_mid"] + h["L_top"], 1),
            "plain_module": plain_mod.astype(float)}


def _powers(a, n):
    out = [np.eye(len(a))]
    for _ in range(n):
        out.append(a @ out[-1])
    return out


def closure_exact(a, w, h):
    c = stationary_cov(a)
    ap = _powers(a, h + LAGS)

    def gam(tau):                       # Cov(x_{t+tau}, x_t)
        return ap[tau] @ c if tau >= 0 else (ap[-tau] @ c).T

    cov_dp = np.hstack([w.T @ (gam(h + j) - gam(j)) @ w for j in range(LAGS)])
    cov_pp = np.block([[w.T @ gam(j - i) @ w for j in range(LAGS)] for i in range(LAGS)])
    v = w.T @ (2 * c - gam(h) - gam(h).T) @ w
    p_macro = v - cov_dp @ np.linalg.solve(cov_pp, cov_dp.T)
    p_micro = w.T @ sum(ap[s] @ ap[s].T for s in range(h)) @ w
    return ratio_gap(v, p_micro, p_macro)


def share_h_exact(a, a0, w, h):
    c = stationary_cov(a)
    ah, a0h = np.linalg.matrix_power(a, h), np.linalg.matrix_power(a0, h)
    p_full = w.T @ sum(m @ m.T for m in _powers(a, h)[:h]) @ w
    dd = ah - a0h
    v = w.T @ (2 * c - ah @ c - c @ ah.T) @ w
    return ratio_gap(v, p_full, p_full + w.T @ dd @ c @ dd.T @ w)


def design():
    h = hierarchy(np.random.default_rng(SEED))
    t = targets(h)
    a, a0 = h["A"], h["A0"]
    d = a - a0
    w, s = gen_spectrum_exact(np.eye(len(a)), np.eye(len(a)) + d @ stationary_cov(a) @ d.T)
    return {"S_exact": s.tolist(), "cuts_exact": layer_cuts(s),
            "overlap": {k: overlap(w[:, -k:], t[name]) for k, name in ((4, "module"), (2, "super"), (1, "global"))},
            "G_module": [closure_exact(a, t["module"], hz) for hz in HORIZONS],
            "G_plain_module": [closure_exact(a, t["plain_module"], hz) for hz in HORIZONS],
            "G_top4": [closure_exact(a, w[:, :4], hz) for hz in HORIZONS],
            "Sh_module": [share_h_exact(a, a0, t["module"], hz) for hz in HORIZONS],
            "Sh_super": [share_h_exact(a, a0, t["super"], hz) for hz in HORIZONS],
            "Sh_global": [share_h_exact(a, a0, t["global"], hz) for hz in HORIZONS]}


def run_hierarchy(rng):
    h = hierarchy(np.random.default_rng(SEED))
    t = targets(h)
    a, a0 = h["A"], h["A0"]
    n = len(a)
    x = np.zeros((T_STEPS + BURN + 1, n))
    noise = rng.standard_normal(x.shape)
    for k in range(len(x) - 1):
        x[k + 1] = a @ x[k] + noise[k]
    x = x[BURN:]
    e, e0 = x[1:] - x[:-1] @ a.T, x[1:] - x[:-1] @ a0.T
    fit, mid, test = thirds(len(e))
    w, _ = gen_spectrum(e[fit], e0[fit])
    s_test = share(e[test], e0[test], w)
    d = a - a0
    _, s_exact = gen_spectrum_exact(np.eye(n), np.eye(n) + d @ stationary_cov(a) @ d.T)
    cuts = layer_cuts(s_test)
    hmax = max(HORIZONS)
    rows_fit = np.arange(mid.start + LAGS, mid.stop - hmax)
    rows_eval = np.arange(test.start + LAGS, len(x) - hmax - 1)
    pw = {hz: np.linalg.matrix_power(a, hz) for hz in HORIZONS}
    pw0 = {hz: np.linalg.matrix_power(a0, hz) for hz in HORIZONS}

    def block_stats(wb):
        y = x @ wb
        g, sh = [], []
        for hz in HORIZONS:
            dy = y[rows_eval + hz] - y[rows_eval]
            v = np.cov(dy.T).reshape(wb.shape[1], -1)
            em = y[rows_eval + hz] - x[rows_eval] @ (wb.T @ pw[hz]).T
            et = y[rows_eval + hz] - x[rows_eval] @ (wb.T @ pw0[hz]).T
            ea = macro_errors(y, rows_fit, rows_eval, hz)
            g.append(ratio_gap(v, em.T @ em / len(em), ea.T @ ea / len(ea)))
            sh.append(ratio_gap(v, em.T @ em / len(em), et.T @ et / len(et)))
        return g, sh

    blocks = {"bottom1": w[:, -1:], "bottom2": w[:, -2:], "bottom4": w[:, -4:], "top4": w[:, :4],
              "plain_module": t["plain_module"]}
    stats = {name: block_stats(wb) for name, wb in blocks.items()}
    profile = {f"bottom{k}": block_stats(w[:, -k:]) for k in range(1, n)}
    return {"S_test": s_test.tolist(), "S_exact": s_exact.tolist(), "cuts": cuts, "w": w.tolist(),
            "overlap": {"module": overlap(w[:, -4:], t["module"]), "super": overlap(w[:, -2:], t["super"]),
                        "global": overlap(w[:, -1:], t["global"])},
            "G": {k: v[0] for k, v in stats.items()}, "Sh": {k: v[1] for k, v in stats.items()},
            "G_profile": {k: v[0] for k, v in profile.items()}, "Sh_profile": {k: v[1] for k, v in profile.items()},
            "design": design(), "module": h["module"].tolist(), "sup": h["sup"].tolist(),
            "target_module": t["module"].tolist(), "target_super": t["super"].tolist(),
            "target_global": t["global"].tolist()}


# ---------------------------------------------------------------- modular Kuramoto
K_IN, K_OUT, SIGMA, DT = 2.0, 0.2, 0.3, 0.05
N_MOD, M_OSC = 3, 4


def kuramoto_setup(rng):
    n = N_MOD * M_OSC
    module = np.arange(n) // M_OSC
    omega = np.array([-0.05, 0.0, 0.05])[module] + 0.2 * rng.standard_normal(n)
    same = module[:, None] == module[None, :]
    kmat = np.where(same, K_IN / (M_OSC - 1), K_OUT / (n - M_OSC))
    np.fill_diagonal(kmat, 0.0)
    return module, omega, kmat


def kuramoto_drift(th, omega, kmat):
    """th: (..., n). Returns omega_i + sum_j k_ij sin(th_j - th_i)."""
    return omega + (kmat * np.sin(th[..., None, :] - th[..., :, None])).sum(-1)


def psi(th):
    return np.concatenate([np.cos(th), np.sin(th)], -1)


def ensemble(th0, omega, kmat, rng, coupled=True):
    """Ensemble mean and covariance of psi at each horizon, from starting states th0 (s, n)."""
    s, n = th0.shape
    th = np.repeat(th0[:, None, :], ENS, 1)
    out, hset = {}, set(HORIZONS)
    for step in range(1, max(HORIZONS) + 1):
        drift = kuramoto_drift(th, omega, kmat) if coupled else omega
        th = th + DT * drift + rng.standard_normal(th.shape) * SIGMA * np.sqrt(DT)
        if step in hset:
            p = psi(th)
            mu = p.mean(1)
            dev = p - mu[:, None, :]
            out[step] = (mu, np.einsum("sri,srj->sij", dev, dev) / (ENS - 1))
    return out


def run_kuramoto(rng):
    module, omega, kmat = kuramoto_setup(rng)
    n = len(omega)
    th = np.zeros((T_STEPS + BURN + 1, n))
    th[0] = rng.uniform(-np.pi, np.pi, n)
    noise = rng.standard_normal(th.shape) * SIGMA * np.sqrt(DT)
    for t in range(len(th) - 1):
        th[t + 1] = th[t] + DT * kuramoto_drift(th[t], omega, kmat) + noise[t]
    th = th[BURN:]
    prev, nxt = th[:-1], th[1:]
    damp = np.exp(-SIGMA ** 2 * DT / 2)

    def expect(mu):
        return np.concatenate([np.cos(mu), np.sin(mu)], 1) * damp

    p_all = psi(th)
    e = p_all[1:] - expect(prev + DT * kuramoto_drift(prev, omega, kmat))
    e0 = p_all[1:] - expect(prev + DT * omega)
    fit, mid, test = thirds(len(e))
    w, _ = gen_spectrum(e[fit], e0[fit])
    s_test = share(e[test], e0[test], w)
    target = np.zeros((2 * n, 2 * N_MOD))
    for m in range(N_MOD):
        target[:n, m] = module == m
        target[n:, N_MOD + m] = module == m
    hmax = max(HORIZONS)
    rows_fit = np.arange(mid.start + LAGS, mid.stop - hmax)
    starts = np.linspace(test.start + LAGS, len(th) - hmax - 2, STARTS).astype(int)
    ens_full = ensemble(th[starts], omega, kmat, np.random.default_rng(SEED + 10))
    ens_twin = ensemble(th[starts], omega, kmat, np.random.default_rng(SEED + 11), coupled=False)

    def block_stats(wb):
        y = p_all @ wb
        g, sh = [], []
        for hz in HORIZONS:
            truth = y[starts + hz]
            dy = truth - y[starts]
            v = np.cov(dy.T).reshape(wb.shape[1], -1)
            mu, cov = ens_full[hz]
            em = truth - mu @ wb
            p_micro = em.T @ em / len(em) - np.einsum("ki,sij,jl->kl", wb.T, cov, wb) / len(em) / ENS
            mu0, cov0 = ens_twin[hz]
            et = truth - mu0 @ wb
            p_twin = et.T @ et / len(et) - np.einsum("ki,sij,jl->kl", wb.T, cov0, wb) / len(et) / ENS
            ea = macro_errors(y, rows_fit, starts, hz, quad=True)
            g.append(ratio_gap(v, p_micro, ea.T @ ea / len(ea)))
            sh.append(ratio_gap(v, p_micro, p_twin))
        return g, sh

    rr = np.random.default_rng(SEED + 12)
    rand = [block_stats(np.linalg.qr(rr.standard_normal((2 * n, 6)))[0]) for _ in range(N_RANDOM)]
    stats = {"bottom6": block_stats(w[:, -6:]), "top6": block_stats(w[:, :6]), "target": block_stats(target)}
    r = np.abs(np.exp(1j * th[test]).reshape(-1, N_MOD, M_OSC).mean(2))
    wb = w[:, -6:]
    dy1 = (p_all[1:] - p_all[:-1])[test] @ wb
    eb, eb0 = e[test] @ wb, e0[test] @ wb
    return {"S_test": s_test.tolist(), "cuts": layer_cuts(s_test), "w": w.tolist(),
            "overlap": overlap(wb, target),
            "S1_block": ratio_gap(np.cov(dy1.T), eb.T @ eb / len(eb), eb0.T @ eb0 / len(eb0)),
            "G": {k: v[0] for k, v in stats.items()}, "Sh": {k: v[1] for k, v in stats.items()},
            "G_random": [g for g, _ in rand], "Sh_random": [s for _, s in rand],
            "module_order": r.mean(0).tolist(), "omega": omega.tolist(), "module": module.tolist()}


# ---------------------------------------------------------------- analysis
def predictions(res):
    hr, ku = res["hierarchy"], res["kuramoto"]
    gap = float(np.max(np.abs(np.array(hr["S_test"]) - np.array(hr["S_exact"]))))
    ov = hr["overlap"]
    g4, g2 = hr["G"]["bottom4"], hr["G"]["bottom2"]
    sh = hr["Sh"]
    med_rand = np.median(np.array(ku["G_random"]), 0)
    gb, gt = np.array(ku["G"]["bottom6"]), np.array(ku["G"]["top6"])
    return [
        ("H1", "hierarchy: held-out spectrum within 0.02 of exact", gap <= 0.02, f"max gap {gap:.4f}"),
        ("H2", "hierarchy: automatic rule finds layers k = 1, 2, 4", hr["cuts"] == [1, 2, 4],
         f"found {hr['cuts']}"),
        ("H3", "hierarchy: discovered layers = true layers (overlap >= 0.98)",
         min(ov.values()) >= 0.98, f"module {ov['module']:.4f}, super {ov['super']:.4f}, global {ov['global']:.4f}"),
        ("H4", "hierarchy: discovered layers closed (G <= 0.02, all h)", max(g4 + g2) <= 0.02,
         "module " + " ".join(f"{x:+.4f}" for x in g4) + "; super " + " ".join(f"{x:+.4f}" for x in g2)),
        ("H5", "hierarchy: plain averages and top-4 not closed",
         hr["G"]["plain_module"][0] >= 0.15 and hr["G"]["top4"][0] >= 0.3,
         f"G(1) plain {hr['G']['plain_module'][0]:.3f}, top-4 {hr['G']['top4'][0]:.3f}"),
        ("H6", "hierarchy: layers emergent, global inherited",
         sh["bottom4"][2] >= 0.1 and sh["bottom2"][3] >= 0.04 and max(abs(x) for x in sh["bottom1"]) <= 0.01,
         f"S_20(module) {sh['bottom4'][2]:.3f}, S_100(super) {sh['bottom2'][3]:.3f}, "
         f"max |S_h(global)| {max(abs(x) for x in sh['bottom1']):.4f}"),
        ("K1", "Kuramoto: bottom-6 = module order parameters (overlap >= 0.9)", ku["overlap"] >= 0.9,
         f"overlap {ku['overlap']:.4f}"),
        ("K2", "Kuramoto: that block is closed (G <= 0.05, all h)", gb.max() <= 0.05,
         "G " + " ".join(f"{x:+.3f}" for x in gb)),
        ("K3", "Kuramoto: more closed than top-6 and median random, all h",
         bool(np.all(gb < gt) and np.all(gb < med_rand)),
         "top-6 " + " ".join(f"{x:.3f}" for x in gt) + "; random " + " ".join(f"{x:.3f}" for x in med_rand)),
        ("K4", "Kuramoto: one-step S <= 0.05 but S_h(100) >= 0.05",
         ku["S1_block"] <= 0.05 and ku["Sh"]["bottom6"][3] >= 0.05,
         f"S_1 {ku['S1_block']:.4f}, S_100 {ku['Sh']['bottom6'][3]:.3f}"),
    ]


def analyze(res) -> str:
    hr, ku = res["hierarchy"], res["kuramoto"]
    hs = " ".join(f"h={h}" for h in HORIZONS)
    lines = ["Experiment 47: closed layers (are the discovered collective variables a level of description?)", "",
             "  Hierarchy (12 walkers, 2 x 2 x 3): held-out S, most blind first: "
             + " ".join(f"{x:.4f}" for x in hr["S_test"][::-1]),
             f"      layer cuts found: {hr['cuts']}",
             f"      G and S_h at {hs}:"]
    for name in ("bottom1", "bottom2", "bottom4", "plain_module", "top4"):
        lines.append(f"        {name:13} G " + " ".join(f"{x:+.4f}" for x in hr["G"][name])
                     + "   S_h " + " ".join(f"{x:+.4f}" for x in hr["Sh"][name]))
    lines += ["", "  Modular Kuramoto (3 x 4, dictionary cos/sin): held-out S, most blind first: "
              + " ".join(f"{x:.4f}" for x in ku["S_test"][::-1][:10]) + " ...",
              f"      layer cuts found: {ku['cuts']}; module order parameters r = "
              + " ".join(f"{x:.2f}" for x in ku["module_order"]),
              f"      G and S_h at {hs}:"]
    for name in ("bottom6", "target", "top6"):
        lines.append(f"        {name:13} G " + " ".join(f"{x:+.4f}" for x in ku["G"][name])
                     + "   S_h " + " ".join(f"{x:+.4f}" for x in ku["Sh"][name]))
    lines.append("        random (median) G " + " ".join(f"{x:+.4f}" for x in np.median(ku["G_random"], 0)))
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    rng = np.random.default_rng(SEED)
    res = {"hierarchy": run_hierarchy(rng), "kuramoto": run_kuramoto(rng), "horizons": list(HORIZONS)}
    (out_dir / "e47_closed_layers_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e47_figures import make_figures
    make_figures(res, out_dir / "e47_figures", text)
    return text


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["design"]:
        np.set_printoptions(precision=4, suppress=True, linewidth=150)
        for key, val in design().items():
            print(key, np.round(np.array(list(val.values()) if isinstance(val, dict) else val), 4))
    else:
        main()
