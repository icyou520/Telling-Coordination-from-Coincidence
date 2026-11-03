"""Primes on stretches of the number line, counted in blocks (e84).

A block is B = 30030 = 2*3*5*7*11*13 consecutive integers starting at a multiple of B, so every block has the same
pattern of divisibility by 2-13 and the same number of integers ending in 1, 3, 7 and 9 (every prime above 5 ends in
one of these). For each block and final digit:
    count     primes, from an exact segmented sieve;
    surv[P]   survivors of trial division by every prime <= P (the cheap filter a search applies before any
              primality test);
    expected  surv[P] / (ln(middle of the block) * prod_{p<=P} (1 - 1/p)), the prime number theorem's count among
              the survivors.
"""

from __future__ import annotations

from math import isqrt, log

import numpy as np

B = 30030
DIGITS = (1, 3, 7, 9)
FILTERS = (101, 1000)


def primes_upto(n: int) -> np.ndarray:
    s = np.ones(n + 1, bool)
    s[:2] = False
    for p in range(2, isqrt(n) + 1):
        if s[p]:
            s[p * p:: p] = False
    return np.flatnonzero(s)


def strike(a: np.ndarray, x0: int, ps: np.ndarray, marks: int = 1 << 23) -> None:
    """Mark multiples of every p in ps within [x0, x0 + len(a)) as False (x0 > max(ps)); large primes in groups of
    about `marks` multiples."""
    n = len(a)
    small = ps[ps < 1 << 15]
    for p in small.tolist():
        a[(-x0) % p:: p] = False
    big = ps[ps >= 1 << 15].astype(np.int64)
    load = np.cumsum(n / big + 1)
    cuts = np.unique(np.concatenate([[0], np.searchsorted(load, np.arange(marks, load[-1], marks)), [len(big)]])) \
        if len(big) else []
    for i, j in zip(cuts[:-1], cuts[1:]):
        p = big[i:j]
        start = (-x0) % p
        k = np.maximum((n - 1 - start) // p + 1, 0)
        tot = int(k.sum())
        if tot == 0:
            continue
        rep = np.repeat(np.arange(len(p)), k)
        step = np.arange(tot) - np.repeat(np.cumsum(k) - k, k)
        a[start[rep] + step * p[rep]] = False


def is_prime(x0: int, n: int) -> np.ndarray:
    """Exact primality of x0 .. x0 + n - 1 (requires x0 > sqrt(x0 + n))."""
    r = isqrt(x0 + n - 1)
    assert x0 > r
    a = np.ones(n, bool)
    strike(a, x0, primes_upto(r))
    return a


def survivors(x0: int, n: int, limit: int) -> np.ndarray:
    a = np.ones(n, bool)
    strike(a, x0, primes_upto(limit))
    return a


def by_digit(a: np.ndarray, blocks: int) -> np.ndarray:
    """(blocks, 4) counts of True per block and final digit 1, 3, 7, 9 (x0 a multiple of B)."""
    return a.reshape(blocks, B // 10, 10).sum(1)[:, list(DIGITS)].astype(np.int64)


def start_at(x: float) -> int:
    return -(-int(x) // B) * B


def panel(x: float, blocks: int, filters=FILTERS) -> dict:
    x0 = start_at(x)
    n = blocks * B
    out = {"x0": x0, "blocks": blocks, "count": by_digit(is_prime(x0, n), blocks)}
    mid = x0 + B * (np.arange(blocks) + 0.5)
    for lim in filters:
        surv = by_digit(survivors(x0, n, lim), blocks)
        mertens = float(np.prod(1 - 1 / primes_upto(lim)))
        out[f"surv{lim}"] = surv
        out[f"expected{lim}"] = surv / (np.log(mid)[:, None] * mertens)
    out["log_x"] = log(x0)
    return out
