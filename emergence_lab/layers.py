"""Multi-scale analysis: at which coarse-graining scales does a layer predict itself?

A field has shape (B, T, C, *space): B independent runs, T time frames,
C channels, and one or two periodic spatial axes. Coarse-graining at scale s
averages s (or s x s) blocks of cells.

For each scale the *layer ladder* asks two questions of every coarse cell:

* own:    how well do the cell and its nearest neighbours, at this scale,
          predict the cell's next value?            (a local macro law G)
* detail: how much better does it get if we also see the finer sub-blocks
          inside the cell?                          (information hidden below)

A good emergent layer has high ``r2_own`` and a small ``gap`` between the two.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from .closure import fit_held_out


def coarsen(field: np.ndarray, s: int) -> np.ndarray:
    """Block-average the spatial axes of a (B, T, C, *space) field by factor s."""
    if s == 1:
        return field
    d = field.ndim - 3
    lead = field.shape[:3]
    shape = list(lead)
    for n in field.shape[3:]:
        shape += [n // s, s]
    return field.reshape(shape).mean(axis=tuple(4 + 2 * i for i in range(d)))


def _local(z: np.ndarray) -> np.ndarray:
    """Stack each cell with its periodic nearest neighbours along the channel axis."""
    d = z.ndim - 3
    parts = [z]
    for ax in range(3, 3 + d):
        parts += [np.roll(z, 1, axis=ax), np.roll(z, -1, axis=ax)]
    return np.concatenate(parts, axis=2)


def _children(fine: np.ndarray) -> np.ndarray:
    """Group a field at scale s/2 into the 2 (1-D) or 4 (2-D) sub-blocks of each scale-s cell."""
    B, T, C = fine.shape[:3]
    if fine.ndim == 4:
        n = fine.shape[3] // 2
        return fine.reshape(B, T, C, n, 2).transpose(0, 1, 2, 4, 3).reshape(B, T, 2 * C, n)
    n, m = fine.shape[3] // 2, fine.shape[4] // 2
    f = fine.reshape(B, T, C, n, 2, m, 2).transpose(0, 1, 2, 4, 6, 3, 5)
    return f.reshape(B, T, 4 * C, n, m)


def _rows(a: np.ndarray) -> np.ndarray:
    """(B, T, F, *grid) -> (B, T, cells, F)."""
    B, T, F = a.shape[:3]
    return np.moveaxis(a.reshape(B, T, F, -1), 2, 3)


def poly(x: np.ndarray, degree: int) -> np.ndarray:
    feats = [x]
    for k in range(2, degree + 1):
        feats += [
            np.prod(x[:, list(c)], axis=1, keepdims=True)
            for c in combinations_with_replacement(range(x.shape[1]), k)
        ]
    return np.hstack(feats)


def scale_ladder(
    field: np.ndarray,
    scales,
    *,
    degree: int = 2,
    max_samples: int = 30_000,
    train_frac: float = 0.7,
    seed: int = 0,
) -> list[dict]:
    field = np.asarray(field, dtype=float)
    d = field.ndim - 3
    rng = np.random.default_rng(seed)
    out = []
    B, T = field.shape[:2]
    is_train = np.zeros((B, T - 1), dtype=bool)
    if B >= 3:
        is_train[: max(1, round(B * train_frac))] = True
    else:
        is_train[:, : max(1, int((T - 1) * train_frac))] = True

    for s in scales:
        z = coarsen(field, s)
        own = _rows(_local(z)[:, :-1])
        target = _rows(z[:, 1:])
        detail = _rows(_children(coarsen(field, s // 2))[:, :-1]) if s > 1 else None

        flat = lambda a: a.reshape(-1, a.shape[-1])
        train_mask = np.broadcast_to(is_train[:, :, None], own.shape[:3]).ravel()
        idx = rng.permutation(len(train_mask))[:max_samples]
        own, target, train_mask = flat(own)[idx], flat(target)[idx], train_mask[idx]
        tr, te = np.flatnonzero(train_mask), np.flatnonzero(~train_mask)

        r2_own, _ = fit_held_out(poly(own, degree), target, tr, te, rng)
        if detail is not None:
            x_full = poly(np.hstack([own, flat(detail)[idx]]), degree)
            r2_detail, _ = fit_held_out(x_full, target, tr, te, rng)
        else:
            r2_detail = r2_own
        out.append(
            {
                "scale": s,
                "cells": int(np.prod(z.shape[3:])),
                "r2_own": r2_own,
                "r2_detail": max(r2_detail, r2_own),
                "gap": max(0.0, r2_detail - r2_own),
                "compression": 1.0 - 1.0 / s**d,
            }
        )
    return out
