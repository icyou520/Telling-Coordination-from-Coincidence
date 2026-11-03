"""Vicsek flocking: self-propelled agents align with neighbours, plus noise.

State is an (N, 3) array of [x, y, heading]. Low noise gives an ordered,
flocking phase; high noise gives a disordered gas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Vicsek:
    n_agents: int = 100
    density: float = 4.0
    speed: float = 0.1
    radius: float = 1.0
    noise: float = 1.0
    """Heading noise is uniform in [-noise/2, noise/2]; the phase transition is near 2-3."""

    @property
    def box(self) -> float:
        return float(np.sqrt(self.n_agents / self.density))

    def initial(self, rng: np.random.Generator) -> np.ndarray:
        pos = rng.random((self.n_agents, 2)) * self.box
        theta = rng.uniform(-np.pi, np.pi, self.n_agents)
        return np.column_stack([pos, theta])

    def neighbour_sums(self, pos: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sum of sin and cos of headings over each agent's neighbourhood (itself included)."""
        sin, cos = np.sin(theta), np.cos(theta)
        if self.n_agents < 200:
            d = pos[:, None, :] - pos[None, :, :]
            d -= self.box * np.round(d / self.box)
            adj = (d**2).sum(-1) <= self.radius**2
            return adj @ sin, adj @ cos
        tree = cKDTree(np.where(pos >= self.box, 0.0, pos), boxsize=self.box)
        i, j = tree.query_pairs(self.radius, output_type="ndarray").T
        n = self.n_agents
        s_sum = sin + np.bincount(i, sin[j], n) + np.bincount(j, sin[i], n)
        c_sum = cos + np.bincount(i, cos[j], n) + np.bincount(j, cos[i], n)
        return s_sum, c_sum

    def step(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pos, theta = s[:, :2], s[:, 2]
        s_sum, c_sum = self.neighbour_sums(pos, theta)
        theta = np.arctan2(s_sum, c_sum)
        theta = theta + self.noise * (rng.random(self.n_agents) - 0.5)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        pos = (pos + self.speed * np.column_stack([np.cos(theta), np.sin(theta)])) % self.box
        return np.column_stack([pos, theta])

    def simulate(self, n_steps: int, burn_in: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        s = self.initial(rng)
        for _ in range(burn_in):
            s = self.step(s, rng)
        out = np.empty((n_steps, self.n_agents, 3))
        for t in range(n_steps):
            s = self.step(s, rng)
            out[t] = s
        return out


def headings(states: np.ndarray) -> np.ndarray:
    """Unit heading vectors, shape (..., N, 2)."""
    th = states[..., 2]
    return np.stack([np.cos(th), np.sin(th)], axis=-1)


def mean_velocity(states: np.ndarray) -> np.ndarray:
    """Macro variable V: average heading vector of the flock, shape (..., 2)."""
    return headings(states).mean(axis=-2)


def polarization(states: np.ndarray) -> np.ndarray:
    """Order parameter |V| in [0, 1]."""
    return np.linalg.norm(mean_velocity(states), axis=-1)
