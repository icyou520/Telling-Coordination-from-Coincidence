"""Simulations for e54: a cohesive flock in open space and noisy Kuramoto oscillators, each with an optional
external field that pins the group's direction (heading psi0 = 0 / phase psi0 = 0)."""

from __future__ import annotations

import numpy as np

DT = 0.1


def flock(n=30, steps=25000, field=0.0, seed=0, burn=2000, k=6, align=1.5, attract=1.0, cohere=0.5,
          repel=1.0, noise=0.35, speed_sd=0.15, speed_cohere=0.3):
    """Positions (T, n, 2), headings (T, n), speeds (T, n). Topological k-nearest alignment, attraction to
    the k neighbours' centre, weak cohesion to the group's centre, short-range repulsion; heterogeneous
    individual cruising speeds, relaxed towards the neighbours' speed, slower when ahead of the centre."""
    rng = np.random.default_rng(seed)
    v0 = 1.0 + speed_sd * rng.standard_normal(n)
    p = rng.normal(0, 1.5, (n, 2))
    th = rng.normal(0, 0.3, n)
    v = v0.copy()
    out_p, out_th, out_v = (np.empty((steps, n, 2)), np.empty((steps, n)), np.empty((steps, n)))
    for t in range(burn + steps):
        d = p[None, :, :] - p[:, None, :]
        dist = np.hypot(d[..., 0], d[..., 1])
        np.fill_diagonal(dist, np.inf)
        nb = np.argsort(dist, 1)[:, :k]
        rows = np.arange(n)[:, None]
        al = np.sin(th[nb] - th[:, None]).mean(1)
        cen = p[nb].mean(1) - p
        at = np.sin(np.arctan2(cen[:, 1], cen[:, 0]) - th) * np.minimum(np.hypot(cen[:, 0], cen[:, 1]), 1.0)
        g = p.mean(0) - p
        gd = np.hypot(g[:, 0], g[:, 1])
        co = np.sin(np.arctan2(g[:, 1], g[:, 0]) - th) * np.clip(gd / 3.0 - 1.0, 0.0, 1.0)
        close = dist[rows, nb] < 0.6
        away = np.where(close[..., None], -d[rows, nb], 0.0).sum(1)
        rp = np.where(close.any(1), np.sin(np.arctan2(away[:, 1], away[:, 0]) - th), 0.0)
        th = th + DT * (align * al + attract * at + cohere * co + repel * rp + field * np.sin(-th)) \
            + noise * np.sqrt(DT) * rng.standard_normal(n)
        ahead = -(g[:, 0] * np.cos(th) + g[:, 1] * np.sin(th))
        v = v + DT * (0.3 * v0 + 0.7 * v[nb].mean(1) - v - speed_cohere * ahead) \
            + 0.1 * np.sqrt(DT) * rng.standard_normal(n)
        v = np.clip(v, 0.3, None)
        p = p + DT * v[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
        if t >= burn:
            out_p[t - burn], out_th[t - burn], out_v[t - burn] = p, np.angle(np.exp(1j * th)), v
    return out_p, out_th, out_v


def kuramoto(n=30, steps=25000, field=0.0, seed=0, burn=2000, coupling=1.5, freq_sd=0.5, noise=0.4):
    """Phases (T, n). Heterogeneous natural frequencies with zero mean, so psi has no systematic drift."""
    rng = np.random.default_rng(seed)
    om = freq_sd * rng.standard_normal(n)
    om -= om.mean()
    th = rng.uniform(-np.pi, np.pi, n)
    out = np.empty((steps, n))
    for t in range(burn + steps):
        z = np.exp(1j * th).mean()
        th = th + DT * (om + coupling * np.abs(z) * np.sin(np.angle(z) - th) + field * np.sin(-th)) \
            + noise * np.sqrt(DT) * rng.standard_normal(n)
        if t >= burn:
            out[t - burn] = th
    return out
