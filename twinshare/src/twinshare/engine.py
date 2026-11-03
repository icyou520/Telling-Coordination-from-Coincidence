"""The calculation engine: twin, group law, interaction share, memory gap, closure and influence.

Notation. x (T, N): N members on one clock; z (T, p): recorded causes every member sees; the group is the members'
mean. A forecast of step t uses each series' history up to t - h (horizon h), summarized as means over the last w
steps for each window w (the HAR form; windows (1, 5, 22) at h = 1 are yesterday, last week, last month).

  member twin   member i's next value from its own history and the recorded causes (no other member's data)
  group law     the group's next value from the group's history and the recorded causes
  share         S = 1 - MSE(group law) / MSE(group law's regression refitted to the summed twin forecasts)

S is the part of the group's predictability that the members' own histories cannot supply: coordination between
members, or a shared driver that is not recorded and that each member reveals poorly. It cannot tell those two apart.

Every fitted stage is cross-fitted: time is cut into blocks, blocks are dealt to folds, and a test fold is forecast
by models fitted on the other folds only, including the member twins whose summed forecast feeds the last stage
(a twin fitted on the test fold would look better than it is and bias S low). Training rows whose history windows
contain a test target are purged.

The memory gap predicts S without fitting anything: from the members' average autocovariance o(k) and the group's
autocovariance g(k), the loss from forecasting the group with its members' memory instead of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Near-zero penalty. A real ridge penalty (1, the lab's choice with weekday dummies only) biases S upward when a
# recorded cause varies slowly (temperature): it shrinks the cause's day-to-day change, which the twin needs to
# separate a member's own past from the cause's past more than the group law does.
LAM = 1e-3


@dataclass(frozen=True)
class Design:
    horizon: int = 1
    windows: tuple[int, ...] = (1, 5, 22)
    block: int | None = None          # rows per cross-fitting block; None = about 30 blocks
    folds: int = 3
    purge: bool = True
    lam: float = LAM
    lagged_causes: bool = True        # causes also enter over the history windows (see causes())

    @property
    def lead(self) -> int:
        """Rows lost at the start: the longest history window ends h steps back."""
        return self.horizon + max(self.windows) - 1

    def block_size(self, rows: int) -> int:
        """About 30 blocks, but between 2 and 10 times the longest history. Long blocks put test periods far from
        the training periods' levels; on drifting series (market volatility) that inflates S, so the analysis
        also reports S at half and double this length."""
        return self.block or max(min(rows // 30, 10 * self.lead), 2 * self.lead, 10)


# ---------------------------------------------------------------- features and fitting

def history(x: np.ndarray, d: Design) -> np.ndarray:
    """Window means of x (T, ...) for rows t = lead .. T-1: shape (T - lead, ..., len(windows))."""
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
    t = np.arange(d.lead, len(x)) - d.horizon + 1
    return np.stack([(c[t] - c[t - w]) / w for w in d.windows], -1)


def causes(z: np.ndarray, d: Design) -> np.ndarray:
    """Recorded-cause features for rows lead..T-1: the causes at t and, with lagged_causes, their means over the
    same history windows as the series. Without the lagged terms a history window still carries the causes' past
    effects (yesterday's promotion in yesterday's sales), which a member can partly ignore but the group, whose
    own noise averages out, cannot: the twin and the group law then weight the past differently and S is
    inflated with no coordination at all."""
    if z.shape[1] == 0 or not d.lagged_causes:
        return z[d.lead:]
    return np.column_stack([z[d.lead:], history(z, d).reshape(len(z) - d.lead, -1)])


def ridge(x: np.ndarray, y: np.ndarray, train: np.ndarray, lam: float = LAM) -> np.ndarray:
    """Ridge on standardized features (intercept unpenalized), fitted on train rows, predicted for all rows."""
    w, b = ridge_coef(x, y, train, lam)
    return x @ w + b


def _scale(x, train):
    """Training mean and scale; columns constant on the training rows (a weekday dummy's weekly mean) get an
    infinite scale, so they drop out instead of turning rounding noise into huge values."""
    mu, sd = x[train].mean(0), x[train].std(0)
    return mu, np.where(sd > 1e-9 * (np.abs(mu) + 1), sd, np.inf)


def ridge_coef(x, y, train, lam=LAM):
    """Coefficients in raw units: prediction = x @ w + b."""
    mu, sd = _scale(x, train)
    z = np.column_stack([np.ones(int(train.sum())), (x[train] - mu) / sd])
    a = z.T @ z + lam * np.diag([0.0] + [1.0] * x.shape[1])
    beta = np.linalg.solve(a, z.T @ y[train])
    w = beta[1:] / sd
    return w, beta[0] - mu @ w


def ridge_cv(base: np.ndarray, extra: np.ndarray, y: np.ndarray, train: np.ndarray, inner: np.ndarray,
             lam: float = LAM) -> np.ndarray:
    """Ridge with base features (penalty lam) and extra features whose penalty is chosen by blocked cross-validation
    within the training rows (inner: fold label per row). For many extra features (member detail), where an
    unpenalized fit overfits; GCV is no use here because time-series errors are autocorrelated."""
    x = np.column_stack([base, extra])
    mu, sd = _scale(x, train)
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    n, nb = int(train.sum()), base.shape[1]

    def fit(rows, k):
        pen = np.array([0.0] + [lam] * nb + [rows.sum() * 10.0 ** k] * extra.shape[1])
        return np.linalg.solve(z[rows].T @ z[rows] + np.diag(pen), z[rows].T @ y[rows])

    grid = range(-3, 5)
    labels = np.unique(inner[train])
    score = [sum(((y - z @ fit(train & (inner != j), k))[train & (inner == j)] ** 2).sum() for j in labels)
             for k in grid]
    return z @ fit(train, grid[int(np.argmin(score))])


def folds(rows: int, d: Design) -> tuple[np.ndarray, np.ndarray]:
    block = np.arange(rows) // d.block_size(rows)
    return block % d.folds, block


def train_rows(fold: np.ndarray, f: int, d: Design) -> np.ndarray:
    """Rows of the other folds, minus (if purging) rows whose history windows contain a test target."""
    te = fold == f
    tr = ~te
    if d.purge:
        r = np.arange(len(fold))
        c = np.concatenate([[0], np.cumsum(te)])
        lo, hi = np.clip(r - d.lead, 0, len(fold)), np.clip(r - d.horizon + 1, 0, len(fold))
        tr &= (c[hi] - c[lo]) == 0
    return tr


@dataclass
class Errors:
    """Per-row squared errors of the group law (e_g) and of the twin's projection (e_0), with block ids."""
    e_g: np.ndarray
    e_0: np.ndarray
    block: np.ndarray

    @property
    def S(self) -> float:
        return float(1 - self.e_g.sum() / self.e_0.sum())

    def interval(self, rng, n_boot=1000, level=0.90) -> tuple[float, float]:
        """Block bootstrap interval for S."""
        nb = int(self.block.max()) + 1
        bg, b0 = np.bincount(self.block, self.e_g, nb), np.bincount(self.block, self.e_0, nb)
        idx = rng.integers(0, nb, (n_boot, nb))
        boot = 1 - bg[idx].sum(1) / b0[idx].sum(1)
        a = (1 - level) / 2
        lo, hi = np.quantile(boot, [a, 1 - a])
        return float(lo), float(hi)

    def S_over(self, mask: np.ndarray) -> float:
        return float(1 - self.e_g[mask].sum() / self.e_0[mask].sum())


def share(x: np.ndarray, z: np.ndarray, d: Design) -> Errors:
    """Leak-free cross-fitted interaction share of the members' mean."""
    rows = len(x) - d.lead
    fold, block = folds(rows, d)
    h, y = history(x, d), x[d.lead:]
    g = x.mean(1)
    env = causes(z, d)
    xg, yg = np.column_stack([history(g, d), env]), g[d.lead:]
    e_g, e_0 = np.empty(rows), np.empty(rows)
    for f in range(d.folds):
        tr, te = train_rows(fold, f, d), fold == f
        if not te.any():
            continue
        y0 = np.mean([ridge(np.column_stack([h[:, i], env]), y[:, i], tr, d.lam) for i in range(x.shape[1])], 0)
        e_g[te] = (yg - ridge(xg, yg, tr, d.lam))[te] ** 2
        e_0[te] = (yg - ridge(xg, y0, tr, d.lam))[te] ** 2
    return Errors(e_g, e_0, block)


def gain(x: np.ndarray, z: np.ndarray, d: Design, members: list[int] | None = None) -> float:
    """Closure test: how much adding members' own histories (all, or the listed ones) improves the group law.
    1 - MSE(group law + member histories) / MSE(group law), cross-fitted, the member terms' penalty chosen by inner
    blocked cross-validation. Near 0: the group is closed (its own past is enough); positive: individual detail
    predicts where the group goes next."""
    rows = len(x) - d.lead
    fold, block = folds(rows, d)
    inner = (block // d.folds) % 3
    g = x.mean(1)
    base = np.column_stack([history(g, d), causes(z, d)])
    sel = range(x.shape[1]) if members is None else members
    extra = history(x[:, list(sel)], d).reshape(rows, -1)
    yg = g[d.lead:]
    e_b, e_f = 0.0, 0.0
    for f in range(d.folds):
        tr, te = train_rows(fold, f, d), fold == f
        if not te.any():
            continue
        e_b += ((yg - ridge(base, yg, tr, d.lam))[te] ** 2).sum()
        e_f += ((yg - ridge_cv(base, extra, yg, tr, inner, d.lam))[te] ** 2).sum()
    return float(1 - e_f / e_b)


def influence(x: np.ndarray, z: np.ndarray, d: Design) -> np.ndarray:
    """Per member: the closure gain from adding that member's history alone (who leads the group)."""
    return np.array([gain(x, z, d, [i]) for i in range(x.shape[1])])


# ---------------------------------------------------------------- the memory gap

def _toeplitz(c, offsets):
    return c[np.abs(offsets[:, None] - offsets[None, :])]


def gap_design(d: Design):
    """The forecast as weights over time offsets: rows of f are the history windows, y the target."""
    feats = [np.arange(-(d.horizon + w - 1), -d.horizon + 1) for w in d.windows]
    offs = np.arange(min(f.min() for f in feats), 1)
    f = np.array([np.isin(offs, w).astype(float) for w in feats])
    return offs, f, (offs == 0).astype(float)


def memories(x: np.ndarray, maxlag: int) -> tuple[np.ndarray, np.ndarray]:
    """Members' mean autocovariance o(k) and the group's (sum) autocovariance g(k), k = 0..maxlag. Biased
    estimator (each lag divided by T), which keeps both positive definite for very smooth series."""
    u = x - x.mean(0)
    v = u.sum(1)
    t = len(u)
    o = np.array([(u[: t - k] * u[k:]).mean(1).sum() / t for k in range(maxlag + 1)])
    g = np.array([(v[: t - k] * v[k:]).sum() / t for k in range(maxlag + 1)])
    return o, g


def s_pred(o: np.ndarray, g: np.ndarray, d: Design) -> float:
    """Memory gap: the population share implied by the members' memory o and the group's memory g."""
    offs, f, y = gap_design(d)
    k = offs - offs.min()
    mo, mg = _toeplitz(o, k), _toeplitz(g, k)
    beta = np.linalg.solve(f @ mo @ f.T, f @ mo @ y)
    a, b, v = f @ mg @ f.T, f @ mg @ y, y @ mg @ y
    best = v - b @ np.linalg.solve(a, b)
    twin = v - 2 * beta @ b + beta @ a @ beta
    return float(1 - best / twin)


def residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Members with the recorded causes regressed out (in-sample least squares; for the memory gap only)."""
    if z.shape[1] == 0:
        return x - x.mean(0)
    a = np.column_stack([np.ones(len(z)), z])
    return x - a @ np.linalg.lstsq(a, x, rcond=None)[0]


def memory_gap(x: np.ndarray, z: np.ndarray, d: Design) -> dict:
    u = residualize(x, z)
    o, g = memories(u, d.lead)
    return {"S_pred": s_pred(o, g, d), "rho_member": float(o[1] / o[0]), "rho_group": float(g[1] / g[0]),
            "o": o, "g": g}


def shift(x: np.ndarray, rng) -> np.ndarray:
    """Virtual group: each member circularly shifted by its own offset (keeps each member, breaks alignment).
    Offsets are spread over the whole circle (bar 5% at each end). Drawing them from a middle band (a quarter to
    three quarters of the length) moves every member by about half the record, which keeps slow regimes aligned
    across members and reads as coordination: on US industry volatility, 0.04 instead of 0."""
    t, n = x.shape
    m = max(t // 20, 1)
    idx = (np.arange(t)[:, None] + rng.integers(m, t - m + 1, n)[None]) % t
    return x[idx, np.arange(n)[None]]
