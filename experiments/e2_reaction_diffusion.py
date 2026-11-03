"""Experiment 2: chemical self-organisation (Gray-Scott reaction-diffusion).

Compares candidate macro descriptions of a field of ~8000 concentrations and
scores each on the four properties. The question is *which* collective
variables form an autonomous layer, not just whether one exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emergence_lab import (
    EmergenceReport,
    macro_determinism,
    predictive_closure,
    print_reports,
    robustness,
)
from emergence_lab.systems.gray_scott import MACROS, GrayScott, micro_features


def generate(gs: GrayScott, n_traj: int, n_steps: int, every: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = np.stack([gs.initial(rng) for _ in range(n_traj)])
    frames = gs.simulate(x0, n_steps, every)
    return frames.transpose(1, 0, 2, 3, 4)


def perturb(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape[1:]) < 0.05
    x[1][mask] = np.clip(x[1][mask] + 0.1 * rng.standard_normal(mask.sum()), 0, 1)
    return x


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    gs = GrayScott()
    n_traj, n_steps, every = (12, 3000, 30) if quick else (24, 5000, 25)
    print(f"  simulating {n_traj} Gray-Scott trajectories x {n_steps} steps ...")
    traj = generate(gs, n_traj, n_steps, every)

    groups = np.repeat(np.arange(n_traj), traj.shape[1] - 1)
    micro = np.array([micro_features(x) for tr in traj for x in tr[:-1]])
    micro_dim = traj.shape[2] * traj.shape[3] * traj.shape[4]

    rng = np.random.default_rng(1)
    starts = [traj[i, rng.integers(5, traj.shape[1] // 2)] for i in range(min(n_traj, 10))]
    stepper = lambda x, _rng: _advance(gs, x, every)

    reports = []
    for name, C in MACROS.items():
        z_all = np.array([[C(x) for x in tr] for tr in traj])
        z, zn = z_all[:, :-1].reshape(-1, z_all.shape[-1]), z_all[:, 1:].reshape(-1, z_all.shape[-1])
        pc = predictive_closure(z, zn, micro, groups=groups)
        reports.append(
            EmergenceReport(
                system="gray-scott",
                macro=name,
                compression=1 - z.shape[1] / micro_dim,
                autonomy=pc["autonomy"],
                determinism=macro_determinism(z, zn, groups=groups),
                robustness=robustness(starts, stepper, lambda x: np.array(C(x)), perturb, 20),
                extra=pc,
            )
        )
    text = print_reports(reports)
    for r in reports:
        line = (f"    {r.macro:<34} R2 macro-only={r.extra['r2_macro']:.3f}  "
                f"R2 with micro={r.extra['r2_full']:.3f}")
        print(line)
        text += "\n" + line

    if out_dir is not None:
        _plot(traj, out_dir / "e2_gray_scott.png")
        text += f"\n  figure: {out_dir / 'e2_gray_scott.png'}"
    return text


def _advance(gs: GrayScott, x: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        x = gs.step(x)
    return x


def _plot(traj: np.ndarray, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = traj.shape[1]
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.5))
    for j, t in enumerate(np.linspace(0, T - 1, 4).astype(int)):
        ax[0, j].imshow(traj[0, t, 1], cmap="magma", vmin=0, vmax=0.5)
        ax[0, j].set_title(f"V field, frame {t}")
        ax[0, j].axis("off")
    for i in range(min(6, traj.shape[0])):
        ax[1, 0].plot(traj[i, :, 1].mean(axis=(1, 2)))
        ax[1, 1].plot(traj[i, :, 0].mean(axis=(1, 2)))
        ax[1, 2].plot(traj[i, :, 1].std(axis=(1, 2)))
        ax[1, 3].plot(traj[i, :, 1].mean(axis=(1, 2)), traj[i, :, 0].mean(axis=(1, 2)))
    for a, t in zip(ax[1], ["mean V", "mean U", "std V", "phase plane (mean V, mean U)"]):
        a.set_title(t)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
