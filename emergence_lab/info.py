"""Information-theoretic measures used to quantify emergence.

Discrete estimators work on samples of hashable states (integers or rows of
integers). Continuous estimators assume (approximately) Gaussian variables.
"""

from __future__ import annotations

import numpy as np

LN2 = np.log(2.0)


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x)
    return x.reshape(len(x), -1)


def _codes(x: np.ndarray) -> np.ndarray:
    """Map each row of a 2-D array to an integer label."""
    _, inv = np.unique(x, axis=0, return_inverse=True)
    return inv.ravel()


def entropy(*variables) -> float:
    """Joint Shannon entropy, in bits, of one or more aligned discrete samples."""
    codes = _codes(np.hstack([_as_2d(v) for v in variables]))
    p = np.bincount(codes) / len(codes)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information(a, b) -> float:
    """I(A;B) in bits."""
    return entropy(a) + entropy(b) - entropy(a, b)


def conditional_entropy(a, given) -> float:
    """H(A | B) in bits."""
    return entropy(a, given) - entropy(given)


def conditional_mutual_information(a, b, given) -> float:
    """I(A;B | C) in bits."""
    return (
        entropy(a, given) + entropy(b, given) - entropy(a, b, given) - entropy(given)
    )


def synergy_report(a, b, y) -> dict:
    """How much of the information about Y exists only in the combination (A, B).

    Uses the minimum-mutual-information redundancy, under which
    synergy = I(A,B;Y) - max(I(A;Y), I(B;Y)).
    """
    i_a = mutual_information(a, y)
    i_b = mutual_information(b, y)
    i_ab = mutual_information(np.hstack([_as_2d(a), _as_2d(b)]), y)
    return {
        "I(A;Y)": i_a,
        "I(B;Y)": i_b,
        "I(A,B;Y)": i_ab,
        "synergy": i_ab - max(i_a, i_b),
    }


def gaussian_mutual_information(x, y, bias_correct: bool = True) -> float:
    """I(X;Y) in bits under a joint-Gaussian assumption.

    The first-order finite-sample bias, dx*dy / (2n) nats, is subtracted when
    ``bias_correct`` is set; this matters when many small terms are summed.
    """
    x, y = _as_2d(x).astype(float), _as_2d(y).astype(float)
    n, dx = x.shape
    dy = y.shape[1]
    cov = np.atleast_2d(np.cov(np.hstack([x, y]), rowvar=False))
    _, ld_x = np.linalg.slogdet(cov[:dx, :dx])
    _, ld_y = np.linalg.slogdet(cov[dx:, dx:])
    _, ld_xy = np.linalg.slogdet(cov)
    mi = 0.5 * (ld_x + ld_y - ld_xy)
    if bias_correct:
        mi -= dx * dy / (2.0 * n)
    return float(mi / LN2)


def ksg_mutual_information(x, y, k: int = 4, seed: int = 0) -> float:
    """I(X;Y) in bits with the Kraskov-Stoegbauer-Grassberger estimator (algorithm 1).

    Nonparametric: no Gaussian assumption. Variables are standardised and a
    tiny jitter breaks ties. Small negative values mean ~zero information.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    rng = np.random.default_rng(seed)
    x, y = _as_2d(x).astype(float), _as_2d(y).astype(float)
    n = len(x)

    def prep(a):
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        return a + 1e-10 * rng.standard_normal(a.shape)

    x, y = prep(x), prep(y)
    dist, _ = cKDTree(np.hstack([x, y])).query(np.hstack([x, y]), k=k + 1, p=np.inf)
    eps = dist[:, -1] * (1 - 1e-10)
    nx = cKDTree(x).query_ball_point(x, eps, p=np.inf, return_length=True) - 1
    ny = cKDTree(y).query_ball_point(y, eps, p=np.inf, return_length=True) - 1
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(mi / LN2)


def emergence_psi(macro_t, macro_next, parts_t, estimator: str = "gaussian") -> dict:
    """Causal-emergence criterion Psi of Rosas et al. (2020).

        Psi = I(V_t; V_{t+1}) - sum_j I(X^j_t; V_{t+1})

    where V is a macro variable computed from the parts X^j. Psi > 0 is a
    sufficient condition for V to carry predictive information about its own
    future that no individual part carries.

    ``parts_t`` has shape (n_samples, n_parts) or (n_samples, n_parts, d).

    The plain sum counts information shared by all parts once per part, so
    Psi is very conservative when parts are redundant (e.g. aligned agents).
    ``psi_corrected`` adds back (n - 1) copies of the minimum-mutual-information
    redundancy, min_j I(X^j_t; V_{t+1}), as suggested by Rosas et al.

    ``estimator`` is "gaussian" (fast, assumes joint normality) or "ksg"
    (nonparametric k-nearest-neighbour estimator).
    """
    mi = {"gaussian": gaussian_mutual_information, "ksg": ksg_mutual_information}[estimator]
    parts_t = np.asarray(parts_t, dtype=float)
    if parts_t.ndim == 2:
        parts_t = parts_t[:, :, None]
    self_info = mi(macro_t, macro_next)
    part_info = np.array([mi(parts_t[:, j], macro_next) for j in range(parts_t.shape[1])])
    psi = self_info - part_info.sum()
    return {
        "psi": psi,
        "psi_corrected": psi + (len(part_info) - 1) * max(part_info.min(), 0.0),
        "I(V_t;V_t+1)": self_info,
        "sum_I(X_j;V_t+1)": float(part_info.sum()),
        "max_I(X_j;V_t+1)": float(part_info.max()),
    }
