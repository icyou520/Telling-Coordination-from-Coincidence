import itertools

import numpy as np
import pytest

from emergence_lab import (
    closure_gap_bits,
    emergence_psi,
    exact_closure,
    gaussian_mutual_information,
    macro_determinism,
    predictive_closure,
    synergy_report,
)
from emergence_lab.systems import eca


def test_flip_pair_difference_is_closed():
    states = list(itertools.product([False, True], repeat=2))
    res = exact_closure(states, lambda s: (not s[0], not s[1]), lambda s: s[0] != s[1])
    assert res.closed
    assert res.macro_law == {False: False, True: True}


@pytest.mark.parametrize(
    "C, closed",
    [(lambda x: x % 2, True), (lambda x: x % 4, True), (lambda x: bin(x).count("1"), False)],
)
def test_counter_macros(C, closed):
    F = lambda x: (x + 1) % 8
    res = exact_closure(range(8), F, C)
    assert res.closed is closed
    z = np.array([C(x) for x in range(8)])
    zn = np.array([C(F(x)) for x in range(8)])
    assert (closure_gap_bits(z, zn) < 1e-12) is closed


def test_xor_is_purely_synergistic():
    a, b = np.array(list(itertools.product([0, 1], repeat=2))).T
    rep = synergy_report(a, b, a ^ b)
    assert rep["I(A;Y)"] == pytest.approx(0)
    assert rep["I(B;Y)"] == pytest.approx(0)
    assert rep["synergy"] == pytest.approx(1)


def test_density_closed_for_traffic_rule_not_rule_30():
    x = eca.all_states(10)
    gap = lambda r: closure_gap_bits(x.sum(1), eca.step(x, r).sum(1))
    assert gap(184) == pytest.approx(0)
    assert gap(30) > 1


def test_rule_105_coarse_grains_to_rule_150():
    """Known result of Israeli & Goldenfeld (2004)."""
    cgs = eca.find_coarse_grainings(105, 2)
    assert {c.macro_rule for c in cgs} == {150}
    assert all(eca.verify_on_lattice(c, 400, 60) == 0 for c in cgs)


def test_rule_30_has_no_small_nontrivial_coarse_graining():
    assert not [c for b in (2, 3) for c in eca.find_coarse_grainings(30, b) if not c.trivial]


def test_gaussian_mi_matches_closed_form():
    rng = np.random.default_rng(0)
    rho = 0.8
    x = rng.standard_normal(200_000)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(200_000)
    assert gaussian_mutual_information(x, y) == pytest.approx(-0.5 * np.log2(1 - rho**2), abs=0.01)


def test_ksg_mi_matches_closed_form_and_detects_independence():
    from emergence_lab.info import ksg_mutual_information

    rng = np.random.default_rng(1)
    rho = 0.8
    x = rng.standard_normal(5000)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(5000)
    assert ksg_mutual_information(x, y) == pytest.approx(-0.5 * np.log2(1 - rho**2), abs=0.05)
    assert abs(ksg_mutual_information(x, rng.standard_normal(5000))) < 0.03


def test_psi_positive_for_synergistic_macro():
    """V = sum of parts follows its own AR(1) law; each part is swamped by noise that cancels in the sum."""
    rng = np.random.default_rng(0)
    n, T = 8, 20_000
    X = np.zeros((T, n))
    for t in range(1, T):
        u = rng.standard_normal(n)
        X[t] = (0.9 * X[t - 1].sum() + 0.3 * rng.standard_normal()) / n + u - u.mean()
    V = X.sum(1, keepdims=True)
    assert emergence_psi(V[:-1], V[1:], X[:-1])["psi"] > 0


def test_psi_positive_for_independent_persistent_parts():
    """Psi > 0 needs no interaction: independent AR(1) parts already satisfy it."""
    rng = np.random.default_rng(0)
    K, T, a = 8, 20_000, 0.99
    X = np.zeros((T, K))
    for t in range(1, T):
        X[t] = a * X[t - 1] + np.sqrt(1 - a**2) * rng.standard_normal(K)
    V = X.mean(1, keepdims=True)
    assert emergence_psi(V[:-1], V[1:], X[:-1])["psi"] > 0.5


def test_closure_detects_hidden_variable():
    """Macro z is driven by a hidden micro variable h: macro alone cannot be autonomous."""
    rng = np.random.default_rng(0)
    T = 4000
    h = np.sin(np.arange(T) * 0.07) + 0.1 * rng.standard_normal(T)
    z = np.cumsum(0.5 * h + 0.05 * rng.standard_normal(T))
    micro = np.column_stack([h, rng.standard_normal((T, 5))])
    pc = predictive_closure(z[:-1, None], z[1:, None], micro[:-1], n_micro_components=6)
    assert pc["closure_gap"] > 0.5
    assert pc["autonomy"] < 0.5


def test_scale_ladder_micro_exact_and_conserved_density_closed():
    from emergence_lab.layers import scale_ladder

    rng = np.random.default_rng(0)
    field = np.stack([eca.run(rng.integers(0, 2, 240, dtype=np.uint8), 184, 100) for _ in range(4)])
    rows = scale_ladder(field[:, :, None, :].astype(float), [1, 240], degree=3)
    assert rows[0]["r2_own"] > 0.99
    assert rows[1]["r2_own"] > 0.99


def test_determinism_high_for_deterministic_map():
    rng = np.random.default_rng(0)
    z = rng.random(3000)
    assert macro_determinism(z, 3.9 * z * (1 - z), predict_change=False) > 0.95
    assert macro_determinism(z, rng.random(3000), predict_change=False) < 0.1
