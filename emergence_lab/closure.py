"""Tests of the cross-scale consistency equation  C(F(x)) = G(C(x)).

F is the micro dynamics, C the coarse-graining, G the candidate macro law.

* ``exact_closure`` checks the condition  C(x) = C(y)  =>  C(F(x)) = C(F(y))
  by enumeration. When it holds, G exists and is returned as a lookup table.
* ``closure_gap_bits`` measures by how much the condition fails, in bits.
* ``predictive_closure`` and ``macro_determinism`` are approximate,
  data-driven versions for continuous and stochastic systems.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable

import numpy as np
from scipy.spatial import cKDTree

from .info import conditional_entropy, entropy


@dataclass
class ExactClosure:
    closed: bool
    macro_law: dict = field(default_factory=dict)
    """z -> G(z) for every macro state whose successor is uniquely determined."""
    conflicts: dict = field(default_factory=dict)
    """z -> set of distinct successors, for macro states where closure fails."""
    witnesses: dict = field(default_factory=dict)
    """z -> one micro state per conflicting successor (counterexamples)."""

    def __str__(self) -> str:
        if self.closed:
            law = ", ".join(f"{z}->{g}" for z, g in sorted(self.macro_law.items(), key=str))
            return f"CLOSED  G = {{{law}}}"
        bad = ", ".join(
            f"{z}->{sorted(s, key=str)}" for z, s in sorted(self.conflicts.items(), key=str)
        )
        return f"NOT CLOSED  ambiguous macro states: {bad}"


def exact_closure(
    states: Iterable,
    F: Callable,
    C: Callable[..., Hashable],
) -> ExactClosure:
    """Enumerate micro states and decide whether an exact macro law G exists."""
    successors: dict = defaultdict(dict)
    for x in states:
        successors[C(x)].setdefault(C(F(x)), x)
    law, conflicts, witnesses = {}, {}, {}
    for z, nxt in successors.items():
        if len(nxt) == 1:
            law[z] = next(iter(nxt))
        else:
            conflicts[z] = set(nxt)
            witnesses[z] = dict(nxt)
    return ExactClosure(not conflicts, law, conflicts, witnesses)


def closure_gap_bits(z, z_next) -> float:
    """H(Z_{t+1} | Z_t) in bits.

    For deterministic micro dynamics this equals I(Z_{t+1}; X_t | Z_t): the
    information about the macro future that lives only in micro details. It is
    zero exactly when the coarse-graining is closed on the sampled states.
    """
    return conditional_entropy(z_next, z)


def compression_bits(x, z) -> float:
    """Fraction of micro entropy discarded by the coarse-graining, 1 - H(Z)/H(X)."""
    hx = entropy(x)
    return 1.0 - entropy(z) / hx if hx > 0 else 0.0


def _poly_features(z: np.ndarray, degree: int) -> np.ndarray:
    feats = [z]
    if degree >= 2:
        iu = np.triu_indices(z.shape[1])
        feats.append((z[:, :, None] * z[:, None, :])[:, iu[0], iu[1]])
    return np.hstack(feats)


def _ridge_predict(xtr, ytr, xte, alpha: float) -> np.ndarray:
    mu, sd = xtr.mean(0), xtr.std(0)
    live = sd > 1e-9 * max(1.0, float(np.abs(xtr).max(initial=0.0)))
    xtr, xte = (xtr[:, live] - mu[live]) / sd[live], (xte[:, live] - mu[live]) / sd[live]
    ym = ytr.mean(0)
    a = xtr.T @ xtr + alpha * len(xtr) * np.eye(xtr.shape[1])
    w = np.linalg.solve(a, xtr.T @ (ytr - ym))
    return xte @ w + ym


def _r2(y, pred) -> float:
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean(0)) ** 2).sum()
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _ridge_r2(xtr, ytr, xte, yte, alpha: float) -> float:
    return _r2(yte, _ridge_predict(xtr, ytr, xte, alpha))


def fit_held_out(x, y, tr, te, rng, alphas=tuple(10.0 ** np.arange(-5, 3)), val_frac=0.3):
    """Ridge regression with the penalty picked on a random validation split of ``tr``.

    Returns (R^2 on ``te``, predictions on ``te``). R^2 is measured against the
    variance of ``y`` over all samples, so a held-out set in which the target
    happens to be constant (e.g. a conserved quantity) is still scored.
    """
    perm = rng.permutation(tr)
    n_val = max(1, int(len(perm) * val_frac))
    val, fit = perm[:n_val], perm[n_val:]
    best = max(alphas, key=lambda a: _ridge_r2(x[fit], y[fit], x[val], y[val], a))
    pred = _ridge_predict(x[tr], y[tr], x[te], best)
    ss_tot = len(te) * y.var(0).sum()
    if ss_tot < 1e-15:
        return 1.0, pred
    return float(1.0 - ((y[te] - pred) ** 2).sum() / ss_tot), pred


def _split(n: int, groups, train_frac: float, rng: np.random.Generator):
    if groups is None:
        cut = int(n * train_frac)
        idx = np.arange(n)
        return idx[:cut], idx[cut:]
    groups = np.asarray(groups)
    uniq = rng.permutation(np.unique(groups))
    train_groups = uniq[: max(1, int(len(uniq) * train_frac))]
    mask = np.isin(groups, train_groups)
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def predictive_closure(
    z,
    z_next,
    micro,
    *,
    groups=None,
    degree: int = 2,
    n_micro_components: int = 20,
    alphas=tuple(10.0 ** np.arange(-5, 3)),
    train_frac: float = 0.7,
    predict_change: bool = True,
    min_r2: float = 0.05,
    seed: int = 0,
) -> dict:
    """Does knowing micro detail improve prediction of the macro future?

    Fits z_{t+1} ~ G(z_t) and z_{t+1} ~ G(z_t, micro_t) with ridge regression
    (penalty chosen on a validation split) and scores both on held-out
    trajectories/blocks.

    ``closure_gap`` = R^2(full) - R^2(macro only): share of the macro future's
    variance explained only by looking below the macro layer.
    ``autonomy`` = R^2(macro only) / best R^2: share of the *predictable* macro
    future that the macro layer captures on its own. NaN when even the micro
    level cannot predict it (R^2 < ``min_r2``): pure noise is not autonomy.

    With ``predict_change`` the target is z_{t+1} - z_t, so slowly varying
    variables cannot score well through persistence alone.
    """
    z, z_next, micro = (np.asarray(a, float).reshape(len(a), -1) for a in (z, z_next, micro))
    if predict_change:
        z_next = z_next - z
    rng = np.random.default_rng(seed)
    tr, te = _split(len(z), groups, train_frac, rng)

    mu = micro[tr].mean(0)
    _, sv, vt = np.linalg.svd(micro[tr] - mu, full_matrices=False)
    keep = min(n_micro_components, int((sv > 1e-3 * sv[0]).sum())) if sv.size and sv[0] > 0 else 0
    pcs = (micro - mu) @ vt[:keep].T

    macro_x = _poly_features(z, degree)
    full_x = np.hstack([macro_x, pcs])
    sub_groups = None if groups is None else np.asarray(groups)[tr]
    fit_tr, val = (tr[i] for i in _split(len(tr), sub_groups, train_frac, rng))

    def held_out(x):
        best = max(alphas, key=lambda a: _ridge_r2(x[fit_tr], z_next[fit_tr], x[val], z_next[val], a))
        pred = _ridge_predict(x[tr], z_next[tr], x[te], best)
        return _r2(z_next[te], pred), pred

    (r2_macro, pred_macro), (r2_full, pred_full) = held_out(macro_x), held_out(full_x)
    ceiling = max(r2_macro, r2_full)
    return {
        "r2_macro": r2_macro,
        "r2_full": r2_full,
        "closure_gap": max(0.0, r2_full - r2_macro),
        "autonomy": float(np.clip(r2_macro / ceiling, 0, 1)) if ceiling > min_r2 else float("nan"),
        "target": z_next[te],
        "pred_macro": pred_macro,
        "pred_full": pred_full,
    }


def macro_determinism(
    z, z_next, *, k: int = 10, groups=None, predict_change: bool = True
) -> float:
    """Model-free, approximate version of  C(x)=C(y) => C(F(x))=C(F(y)).

    For each sample, find the k samples with the most similar macro state
    (from other groups/trajectories when ``groups`` is given, so neighbours
    are genuinely different micro realisations) and measure how much their
    macro successors disagree.

    Returns 1 - E[local variance of z_{t+1}] / Var(z_{t+1}):
    1 means the macro state fully determines its successor, 0 means it
    carries no information about it. ``predict_change`` scores the change
    z_{t+1} - z_t instead of the level, as in ``predictive_closure``.
    """
    z, z_next = (np.asarray(a, float).reshape(len(a), -1) for a in (z, z_next))
    if predict_change:
        z_next = z_next - z
    zs = (z - z.mean(0)) / (z.std(0) + 1e-12)
    tree = cKDTree(zs)
    n_query = min(len(z), (k + 1) * (6 if groups is not None else 1))
    _, nbrs = tree.query(zs, k=n_query)
    if groups is not None:
        groups = np.asarray(groups)
    local = np.empty(len(z))
    for i in range(len(z)):
        cand = nbrs[i][nbrs[i] != i]
        if groups is not None:
            other = cand[groups[cand] != groups[i]]
            cand = other if len(other) >= 2 else cand
        cand = cand[:k]
        local[i] = z_next[cand].var(0).sum() * len(cand) / max(len(cand) - 1, 1)
    total = z_next.var(0).sum()
    return float(1.0 - local.mean() / total) if total > 0 else 1.0


def robustness(
    initial_states,
    step: Callable,
    C: Callable,
    perturb: Callable,
    n_steps: int,
    seed: int = 0,
) -> float:
    """Does the macro trajectory survive changes to individual components?

    Each initial state is run twice with identical randomness, once after a
    micro perturbation. Returns 1 - (macro divergence) / (macro spread across
    unrelated runs): 1 means perturbations are invisible at the macro level,
    0 means a perturbed run is as different as an unrelated one.
    """
    rng = np.random.default_rng(seed)
    finals, diverge = [], []
    for x in initial_states:
        s = int(rng.integers(2**31))
        a, b = x.copy(), perturb(x.copy(), np.random.default_rng(s + 1))
        ra, rb = np.random.default_rng(s), np.random.default_rng(s)
        for _ in range(n_steps):
            a, b = step(a, ra), step(b, rb)
        za, zb = np.atleast_1d(C(a)), np.atleast_1d(C(b))
        finals.append(za)
        diverge.append(((za - zb) ** 2).sum())
    finals = np.array(finals)
    spread = 2.0 * finals.var(0).sum()
    return float(1.0 - np.mean(diverge) / spread) if spread > 0 else 1.0
