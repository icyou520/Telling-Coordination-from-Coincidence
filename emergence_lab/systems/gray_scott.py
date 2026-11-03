"""Gray-Scott reaction-diffusion: U + 2V -> 3V,  V -> P, with feed and kill.

Default parameters are in the 'mitosis' regime, where spots of V grow and
divide until they fill the domain: a chemical system whose collective
structures (spots) replicate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GrayScott:
    size: int = 64
    Du: float = 0.16
    Dv: float = 0.08
    F: float = 0.0367
    k: float = 0.0649
    dt: float = 1.0

    def initial(self, rng: np.random.Generator, n_seeds: int | None = None) -> np.ndarray:
        """State has shape (2, size, size): channel 0 = U, channel 1 = V."""
        n = self.size
        x = np.zeros((2, n, n))
        x[0] = 1.0
        n_seeds = n_seeds if n_seeds is not None else int(rng.integers(1, 6))
        for _ in range(n_seeds):
            i, j = rng.integers(0, n, 2)
            r = int(rng.integers(3, 7))
            rows = (np.arange(i - r, i + r) % n)[:, None]
            cols = (np.arange(j - r, j + r) % n)[None, :]
            x[0, rows, cols] = 0.5
            x[1, rows, cols] = 0.25
        x += 0.02 * rng.standard_normal(x.shape)
        return np.clip(x, 0.0, 1.0)

    def step(self, x: np.ndarray, rng=None) -> np.ndarray:
        """One Euler step; works on (2, n, n) or batched (B, 2, n, n) arrays."""
        u, v = x[..., 0, :, :], x[..., 1, :, :]
        uvv = u * v * v
        du = self.Du * _laplacian(u) - uvv + self.F * (1.0 - u)
        dv = self.Dv * _laplacian(v) + uvv - (self.F + self.k) * v
        return np.stack([u + self.dt * du, v + self.dt * dv], axis=-3)

    def simulate(self, x0: np.ndarray, n_steps: int, record_every: int) -> np.ndarray:
        frames = [x0]
        x = x0
        for t in range(1, n_steps + 1):
            x = self.step(x)
            if t % record_every == 0:
                frames.append(x)
        return np.stack(frames)


def _laplacian(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, -1) + np.roll(a, -1, -1) + np.roll(a, 1, -2) + np.roll(a, -1, -2) - 4.0 * a
    )


def count_spots(v: np.ndarray, threshold: float = 0.2) -> int:
    """Number of connected V-rich regions (periodic boundaries approximated)."""
    _, n = ndimage.label(v > threshold)
    return int(n)


MACROS = {
    "mean V": lambda x: [x[1].mean()],
    "mean U, mean V": lambda x: [x[0].mean(), x[1].mean()],
    "mean U, mean V, std V": lambda x: [x[0].mean(), x[1].mean(), x[1].std()],
    "mean U,V + std V + #spots": lambda x: [
        x[0].mean(),
        x[1].mean(),
        x[1].std(),
        count_spots(x[1]) / 50.0,
    ],
}


def micro_features(x: np.ndarray, grid: int = 16) -> np.ndarray:
    """Block-averaged U and V fields: a high-dimensional stand-in for micro detail."""
    n = x.shape[-1]
    b = n // grid
    return x.reshape(2, grid, b, grid, b).mean(axis=(2, 4)).ravel()
