"""Elementary cellular automata and automatic discovery of exact coarse-grainings.

A coarse-graining (Israeli & Goldenfeld, PRL 2004) groups cells into blocks of
size b, maps each block to a single bit with a projection P, and asks whether
some elementary rule g reproduces b micro steps in one macro step:

    P(f^b(x)) = g(P(x))      for every micro configuration x

This is exactly C(F(x)) = G(C(x)) with C = blockwise P, F = f^b, G = g.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rule_table(rule: int) -> np.ndarray:
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)


def step(x: np.ndarray, rule: int | np.ndarray) -> np.ndarray:
    """One synchronous update on a periodic ring (last axis = space)."""
    table = rule_table(rule) if np.isscalar(rule) else rule
    idx = 4 * np.roll(x, 1, axis=-1) + 2 * x + np.roll(x, -1, axis=-1)
    return table[idx]


def run(x0: np.ndarray, rule: int, n_steps: int) -> np.ndarray:
    table = rule_table(rule)
    out = np.empty((n_steps + 1, x0.size), dtype=np.uint8)
    out[0] = x0
    for t in range(n_steps):
        out[t + 1] = step(out[t], table)
    return out


def all_states(n: int) -> np.ndarray:
    """Every configuration of n bits, shape (2**n, n)."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)


def _bits_to_int(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.int64) << np.arange(bits.shape[-1])).sum(-1)


def depends_on(rule: int) -> tuple[bool, bool, bool]:
    """Which of (left, centre, right) the rule's output actually depends on."""
    t = rule_table(rule)
    return tuple(
        any(t[i] != t[i ^ bit] for i in range(8)) for bit in (4, 2, 1)
    )


@dataclass(frozen=True)
class CoarseGraining:
    rule: int
    block: int
    projection: tuple[int, ...]
    """projection[k] = macro bit for the block whose bits encode integer k (LSB = leftmost)."""
    macro_rule: int

    @property
    def trivial(self) -> bool:
        """Macro rule ignores at least two of its three inputs (constant, identity, shift...)."""
        return sum(depends_on(self.macro_rule)) <= 1

    def coarse(self, x: np.ndarray) -> np.ndarray:
        blocks = x.reshape(*x.shape[:-1], -1, self.block)
        return np.asarray(self.projection, dtype=np.uint8)[_bits_to_int(blocks)]

    def describe(self) -> str:
        ones = [
            "".join(str((k >> i) & 1) for i in range(self.block))
            for k, p in enumerate(self.projection)
            if p
        ]
        return (
            f"rule {self.rule:3d} --(block {self.block}, P=1 on {{{','.join(ones)}}})--> "
            f"rule {self.macro_rule}"
        )


def _block_dynamics(rule: int, b: int):
    """For every 3-block configuration: codes of the left/mid/right blocks and of the
    middle block after b micro steps."""
    table = rule_table(rule)
    cfg = all_states(3 * b)
    y = cfg
    for _ in range(b):
        y = table[4 * y[:, :-2] + 2 * y[:, 1:-1] + y[:, 2:]]
    left, mid, right = (_bits_to_int(cfg[:, i * b : (i + 1) * b]) for i in range(3))
    return left, mid, right, _bits_to_int(y)


def best_balanced_coarse_graining(rule: int, block: int) -> tuple[CoarseGraining, float]:
    """Closest thing to an emergent rule when no exact one exists.

    Searches projections that map exactly half the block patterns to 1, picks
    for each macro neighbourhood the most common macro outcome, and returns
    the coarse-graining with the smallest fraction of configurations where
    C(F(x)) != G(C(x)).
    """
    left, mid, right, centre_out = _block_dynamics(rule, block)
    n_pat = 2**block
    best, best_err = None, 2.0
    for code in range(1, 2**n_pat - 1):
        P = ((code >> np.arange(n_pat)) & 1).astype(np.uint8)
        if P.sum() != n_pat // 2:
            continue
        macro_in = 4 * P[left] + 2 * P[mid] + P[right]
        macro_out = P[centre_out]
        ones = np.bincount(macro_in, weights=macro_out, minlength=8)
        total = np.bincount(macro_in, minlength=8)
        g = (ones * 2 > total).astype(np.int64)
        err = float(np.mean(g[macro_in] != macro_out))
        if err < best_err:
            g_rule = int((g << np.arange(8)).sum())
            best, best_err = CoarseGraining(rule, block, tuple(int(p) for p in P), g_rule), err
    return best, best_err


def neighbourhood_information(rule: int) -> dict:
    """Bits of information each part of the neighbourhood carries about a cell's next state.

    The neighbourhood is drawn uniformly. ``synergy`` is information present
    only in combinations of cells, i.e. I(Y; L,C,R) - max over single cells.
    """
    from ..info import mutual_information

    cfg = all_states(3)
    L, C, R = cfg[:, 2], cfg[:, 1], cfg[:, 0]
    Y = rule_table(rule)[4 * L + 2 * C + R]
    single = {
        "left": mutual_information(L, Y),
        "centre": mutual_information(C, Y),
        "right": mutual_information(R, Y),
    }
    whole = mutual_information(cfg, Y)
    return {**single, "all three": whole, "synergy": whole - max(single.values())}


def find_coarse_grainings(rule: int, block: int) -> list[CoarseGraining]:
    """Search all non-constant projections P: {0,1}^block -> {0,1}.

    After ``block`` micro steps, a block depends only on itself and its two
    neighbouring blocks, so checking all 2**(3*block) local configurations is
    an exact proof for rings of every size divisible by ``block``.
    """
    b = block
    left, mid, right, centre_out = _block_dynamics(rule, b)
    found = []
    for code in range(1, 2 ** (2**b) - 1):
        P = ((code >> np.arange(2**b)) & 1).astype(np.uint8)
        macro_in = 4 * P[left] + 2 * P[mid] + P[right]
        macro_out = P[centre_out]
        seen = np.full(8, -1)
        seen[macro_in] = macro_out
        if (seen < 0).any():
            continue
        if np.all(seen[macro_in] == macro_out):
            g = int((seen.astype(np.int64) << np.arange(8)).sum())
            found.append(CoarseGraining(rule, b, tuple(int(p) for p in P), g))
    return found


def verify_on_lattice(cg: CoarseGraining, n_cells: int, n_macro_steps: int, seed: int = 0) -> int:
    """Simulate both paths of the consistency equation; return number of mismatching cells."""
    rng = np.random.default_rng(seed)
    n_cells -= n_cells % cg.block
    x = rng.integers(0, 2, n_cells, dtype=np.uint8)
    micro_table, macro_table = rule_table(cg.rule), rule_table(cg.macro_rule)
    z = cg.coarse(x)
    mismatches = 0
    for _ in range(n_macro_steps):
        for _ in range(cg.block):
            x = step(x, micro_table)
        z = step(z, macro_table)
        mismatches += int((cg.coarse(x) != z).sum())
    return mismatches
