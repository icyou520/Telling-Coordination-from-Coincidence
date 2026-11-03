"""Systems as seen by the explorer: a live simulation, its layers, and the tests.

Each view owns one simulated system and knows how to draw it at three levels
(micro, a coarse-grained layer at a chosen scale, and the top macro variable),
how to build the multi-scale "layer ladder", and how to run and draw the
four tests.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..closure import macro_determinism, predictive_closure
from ..info import emergence_psi
from ..layers import scale_ladder
from ..systems import eca
from ..systems.gray_scott import GrayScott, micro_features
from ..systems.vicsek import Vicsek, headings, mean_velocity, polarization


class SystemView:
    name = ""
    param_label = ""
    param_range: tuple = (0.0, 1.0, 0.5, None)
    """(min, max, initial, step)"""
    scales: list[int] = []
    scale_unit = "cells"
    ladder_degree = 2
    F_desc = ""
    macro_name = ""

    def __init__(self):
        self.param = self.param_range[2]
        self.seed = 0
        self._ladder: dict = {}
        self._tests: dict = {}
        self.reset()

    def set_param(self, value):
        self.param = value
        self.reset()

    def ladder(self) -> list[dict]:
        if self.param not in self._ladder:
            self._ladder[self.param] = scale_ladder(
                self.ladder_field(), self.scales, degree=self.ladder_degree
            )
        return self._ladder[self.param]

    def ladder_ready(self) -> bool:
        return self.param in self._ladder

    def tests(self) -> dict:
        if self.param not in self._tests:
            self._tests[self.param] = self.compute_tests()
        return self._tests[self.param]

    def tests_ready(self) -> bool:
        return self.param in self._tests

    def C_desc(self, s: int) -> str:
        return f"average each block of {s} {self.scale_unit}"

    def title(self) -> str:
        return f"{self.name}  ({self.param_label} = {self._fmt(self.param)})"

    @staticmethod
    def _fmt(v) -> str:
        return f"{v:g}" if isinstance(v, float) else str(v)

    # subclasses implement:
    def reset(self): ...
    def advance(self): ...
    def perturb(self, rng): ...

    def setup_live(self, ax_micro, ax_layer, ax_macro, s: int) -> list:
        """Create the artists for the three live panels once; return those that change."""

    def update_live(self):
        """Push the current state into the artists created by ``setup_live``."""

    def ladder_field(self) -> np.ndarray: ...
    def compute_tests(self) -> dict: ...
    def draw_tests(self, axes): ...
    test_captions: tuple[str, str, str, str] = ("", "", "", "")


# --------------------------------------------------------------------------- CA


_DENSITY_GAPS: np.ndarray | None = None


def _density_gaps() -> np.ndarray:
    global _DENSITY_GAPS
    if _DENSITY_GAPS is None:
        from ..closure import closure_gap_bits

        x = eca.all_states(10)
        z = x.sum(1)
        _DENSITY_GAPS = np.array(
            [closure_gap_bits(z, eca.step(x, r).sum(1)) for r in range(256)]
        )
    return _DENSITY_GAPS


class CAView(SystemView):
    name = "Cellular automaton"
    param_label = "rule"
    param_range = (0, 255, 105, 1)
    scales = [1, 2, 4, 8, 16, 48, 240]
    ladder_degree = 3
    macro_name = "global density"
    N, H = 240, 120

    @property
    def F_desc(self):
        return f"elementary rule {int(self.param)}: each cell looks at itself and its 2 neighbours"

    def reset(self):
        self.rule = int(self.param)
        self.table = eca.rule_table(self.rule)
        rng = np.random.default_rng(self.seed)
        self.x = rng.integers(0, 2, self.N, dtype=np.uint8)
        self.history = deque(maxlen=self.H)
        self.density = deque(maxlen=300)
        self._record()

    def _record(self):
        self.history.append(self.x.copy())
        self.density.append(self.x.mean())

    def advance(self):
        self.x = eca.step(self.x, self.table)
        self._record()

    def perturb(self, rng):
        idx = (int(rng.integers(self.N)) + np.arange(8)) % self.N
        self.x[idx] ^= 1
        self.history[-1] = self.x.copy()

    def _hist(self) -> np.ndarray:
        h = np.zeros((self.H, self.N))
        arr = np.array(self.history)
        h[-len(arr):] = arr
        return h

    def setup_live(self, ax_micro, ax_layer, ax_macro, s):
        self._s = s
        h = self._hist()
        self._im_micro = ax_micro.imshow(h, cmap="binary", aspect="auto",
                                         interpolation="nearest", vmin=0, vmax=1)
        ax_micro.set_xlabel("cell")
        ax_micro.set_ylabel("time (newest at bottom)")
        self._im_layer = ax_layer.imshow(h.reshape(self.H, -1, s).mean(-1), cmap="viridis",
                                         aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax_layer.set_xlabel(f"block of {s} cells")
        (self._line,) = ax_macro.plot([], [], color="tab:purple")
        ax_macro.set_xlim(0, self.density.maxlen)
        ax_macro.set_ylim(0, 1)
        ax_macro.set_xlabel("time (last 300 steps)")
        ax_macro.set_ylabel("fraction of cells on")
        self.update_live()
        return [self._im_micro, self._im_layer, self._line]

    def update_live(self):
        h = self._hist()
        self._im_micro.set_data(h)
        self._im_layer.set_data(h.reshape(self.H, -1, self._s).mean(-1))
        d = np.array(self.density)
        self._line.set_data(np.arange(len(d)), d)

    def ladder_field(self):
        rng = np.random.default_rng(1)
        runs = [eca.run(rng.integers(0, 2, self.N, dtype=np.uint8), self.rule, 300) for _ in range(4)]
        return np.stack(runs)[:, :, None, :].astype(float)

    test_captions = (
        "CLOSURE (two paths). Path A: simulate every cell, then coarse-grain. Path B: run only "
        "the emergent rule. Black/white = both paths agree, red = they disagree. No red means "
        "C(F(x)) = G(C(x)) exactly: a new layer with its own law.",
        "IS DENSITY AUTONOMOUS? For all 256 rules: bits of the next density that today's density "
        "cannot predict, H(Z'|Z). Zero (green) means density obeys its own law. Red dot = this rule.",
        "ROBUSTNESS. Flip a single cell and show (red) where the perturbed run differs from the "
        "original. Contained damage = robust; spreading damage = sensitive to its parts.",
        "SYNERGY. Bits each neighbour alone tells about a cell's next state, versus all three "
        "together. XOR-like rules (e.g. 90) carry information only in combination.",
    )

    def compute_tests(self):
        r = self.rule
        exact = [c for b in (2, 3) for c in eca.find_coarse_grainings(r, b) if not c.trivial]
        if exact:
            cg, err = exact[0], 0.0
        else:
            cg, err = eca.best_balanced_coarse_graining(r, 2)
        rng = np.random.default_rng(2)
        x0 = rng.integers(0, 2, 120, dtype=np.uint8)
        steps = 50
        path_a = cg.coarse(eca.run(x0, r, steps * cg.block)[:: cg.block])
        path_b = eca.run(cg.coarse(x0), cg.macro_rule, steps)

        x0 = rng.integers(0, 2, self.N, dtype=np.uint8)
        y0 = x0.copy()
        y0[self.N // 2] ^= 1
        a, b = eca.run(x0, r, 120), eca.run(y0, r, 120)
        return {
            "cg": cg, "exact": bool(exact), "err": err, "n_exact": len(exact),
            "path_a": path_a, "path_b": path_b,
            "gaps": _density_gaps(),
            "damage": a != b,
            "info": eca.neighbourhood_information(r),
        }

    def draw_tests(self, axes):
        t = self.tests()
        ax = axes[0]
        a, b = t["path_a"], t["path_b"]
        img = np.ones(a.shape + (3,))
        img[(a == 1) & (b == 1)] = 0.15
        img[a != b] = (0.85, 0.1, 0.1)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        cg = t["cg"]
        if t["exact"]:
            head = (f"EXACT: rule {cg.rule} -> rule {cg.macro_rule} (block {cg.block}), "
                    f"{int((a != b).sum())} mismatches")
        else:
            head = (f"No exact layer at blocks<=3. Best guess: rule {cg.macro_rule}, "
                    f"{100 * (a != b).mean():.0f}% cells wrong")
        ax.set_title(head, fontsize=9)
        ax.set_xlabel("macro cell")
        ax.set_ylabel("macro time")

        ax = axes[1]
        gaps = t["gaps"]
        closed = gaps < 1e-9
        ax.scatter(np.flatnonzero(~closed), gaps[~closed], s=8, color="0.6")
        ax.scatter(np.flatnonzero(closed), gaps[closed], s=14, color="tab:green", label="density closed")
        ax.scatter([self.rule], [gaps[self.rule]], s=90, color="tab:red", zorder=5,
                   label=f"rule {self.rule}: {gaps[self.rule]:.2f} bits")
        ax.set_xlabel("rule")
        ax.set_ylabel("H(next density | density)  [bits]")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("Closure gap of density, all rules", fontsize=9)

        ax = axes[2]
        dmg = t["damage"]
        ax.imshow(dmg, cmap="Reds", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"Flip 1 cell: {100 * dmg[-1].mean():.1f}% of cells differ after "
                     f"{len(dmg) - 1} steps", fontsize=9)
        ax.set_xlabel("cell")
        ax.set_ylabel("time")

        ax = axes[3]
        info = t["info"]
        keys = ["left", "centre", "right", "all three", "synergy"]
        colors = ["tab:blue"] * 3 + ["tab:purple", "tab:orange"]
        ax.bar(keys, [info[k] for k in keys], color=colors)
        ax.set_ylabel("bits about the next state")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Where the information is (rule {self.rule})", fontsize=9)


# --------------------------------------------------------------------- chemistry


class ChemView(SystemView):
    name = "Chemistry"
    param_label = "feed rate F"
    param_range = (0.020, 0.060, 0.0367, 0.0001)
    scales = [1, 2, 4, 8, 16, 32, 64]
    scale_unit = "pixels"
    macro_name = "mean U, mean V"
    F_desc = "Gray-Scott reaction-diffusion: U + 2V -> 3V, V -> waste, both diffuse"
    steps_per_frame = 20

    def C_desc(self, s):
        return f"average U and V over {s}x{s} pixel blocks"

    def reset(self):
        self.gs = GrayScott(F=float(self.param))
        self.x = self.gs.initial(np.random.default_rng(self.seed), n_seeds=3)
        self.trail = deque(maxlen=600)
        self._record()

    def _record(self):
        self.trail.append((self.x[1].mean(), self.x[0].mean()))

    def advance(self):
        for _ in range(self.steps_per_frame):
            self.x = self.gs.step(self.x)
        self._record()

    def perturb(self, rng):
        n = self.gs.size
        i, j = rng.integers(0, n, 2)
        rows, cols = (np.arange(i - 4, i + 4) % n)[:, None], (np.arange(j - 4, j + 4) % n)[None, :]
        self.x[0, rows, cols] = 0.5
        self.x[1, rows, cols] = 0.25

    def _coarse_v(self) -> np.ndarray:
        n = self.gs.size // self._s
        return self.x[1].reshape(n, self._s, n, self._s).mean(axis=(1, 3))

    def setup_live(self, ax_micro, ax_layer, ax_macro, s):
        self._s = s
        self._im_micro = ax_micro.imshow(self.x[1], cmap="magma", vmin=0, vmax=0.5,
                                         interpolation="nearest")
        ax_micro.set_xticks([])
        ax_micro.set_yticks([])
        ax_micro.set_xlabel("concentration of V at every pixel")
        self._im_layer = ax_layer.imshow(self._coarse_v(), cmap="magma", vmin=0, vmax=0.5,
                                         interpolation="nearest")
        ax_layer.set_xticks([])
        ax_layer.set_yticks([])
        n = self.gs.size // s
        ax_layer.set_xlabel(f"{n}x{n} blocks")
        (self._trail_line,) = ax_macro.plot([], [], color="tab:purple", lw=1)
        (self._dot,) = ax_macro.plot([], [], "o", color="tab:red")
        ax_macro.set_xlabel("mean V")
        ax_macro.set_ylabel("mean U")
        ax_macro.set_xlim(0, 0.16)
        ax_macro.set_ylim(0.55, 1.0)
        self.update_live()
        return [self._im_micro, self._im_layer, self._trail_line, self._dot]

    def update_live(self):
        self._im_micro.set_data(self.x[1])
        self._im_layer.set_data(self._coarse_v())
        tr = np.array(self.trail)
        self._trail_line.set_data(tr[:, 0], tr[:, 1])
        self._dot.set_data(tr[-1:, 0], tr[-1:, 1])

    def ladder_field(self):
        rng = np.random.default_rng(1)
        x0 = np.stack([self.gs.initial(rng) for _ in range(6)])
        return self.gs.simulate(x0, 4000, 100).transpose(1, 0, 2, 3, 4)

    test_captions = (
        "AUTONOMY. Predict the next change of (mean U, mean V) from the macro state alone (blue) "
        "or also from a 16x16 map of the field (orange). If orange is no better, the macro layer "
        "is closed: it doesn't need micro detail.",
        "DETERMINISM. Each dot is a moment from one of 8 runs (colour = run). If runs with the "
        "same macro state make the same macro change, the dots form a thin curve: the macro "
        "state fixes its own future.",
        "ROBUSTNESS. Drop extra noise into 5% of the pixels (dashed) and compare with the "
        "original run (solid). Lines that stay together mean the macro pattern survives "
        "changes to its parts.",
        "CAUSAL EMERGENCE (Psi). Does (mean U, mean V) predict its own future better than the "
        "16 regional parts do individually? Psi > 0 means yes. Corrected Psi adds back "
        "information shared by many parts.",
    )

    def compute_tests(self):
        rng = np.random.default_rng(3)
        x0 = np.stack([self.gs.initial(rng) for _ in range(8)])
        traj = self.gs.simulate(x0, 3000, 25).transpose(1, 0, 2, 3, 4)
        B, T = traj.shape[:2]
        z_all = np.stack([traj[:, :, 0].mean((-1, -2)), traj[:, :, 1].mean((-1, -2))], -1)
        z, zn = z_all[:, :-1].reshape(-1, 2), z_all[:, 1:].reshape(-1, 2)
        groups = np.repeat(np.arange(B), T - 1)
        micro = np.array([micro_features(x) for run in traj for x in run[:-1]])
        pc = predictive_closure(z, zn, micro, groups=groups)
        det = macro_determinism(z, zn, groups=groups)

        pairs = []
        for i in range(5):
            x = traj[i, T // 4].copy()
            y = x.copy()
            mask = rng.random(y.shape[1:]) < 0.05
            y[1][mask] = np.clip(y[1][mask] + 0.1 * rng.standard_normal(mask.sum()), 0, 1)
            a_line, b_line = [], []
            for _ in range(60):
                for _ in range(25):
                    x, y = self.gs.step(x), self.gs.step(y)
                a_line.append(x[1].mean())
                b_line.append(y[1].mean())
            pairs.append((np.array(a_line), np.array(b_line)))
        finals = np.array([a[-1] for a, _ in pairs])
        div = np.mean([(a[-1] - b[-1]) ** 2 for a, b in pairs])
        rob = 1 - div / (2 * finals.var()) if finals.var() > 0 else 1.0

        n = traj.shape[-1] // 4
        parts = traj.reshape(B, T, 2, 4, n, 4, n).mean((4, 6)).reshape(B, T, 2, 16)
        parts = np.moveaxis(parts, 2, 3)
        psi = emergence_psi(z, zn, parts[:, :-1].reshape(-1, 16, 2))
        return {"pc": pc, "det": det, "z": z, "dz": zn - z, "groups": groups,
                "pairs": pairs, "rob": rob, "psi": psi, "zlabel": "mean V",
                "zdim": 1, "plabel": "mean V"}

    def draw_tests(self, axes):
        draw_continuous_tests(axes, self.tests())


# ---------------------------------------------------------------------- flocking


class FlockView(SystemView):
    name = "Flocking"
    param_label = "noise"
    param_range = (0.0, 6.0, 2.0, 0.1)
    scales = [1, 2, 4, 8, 16]
    scale_unit = "grid squares"
    macro_name = "polarization |V|"
    F_desc = "Vicsek model: each agent turns toward its neighbours' average heading, plus noise"
    G = 16
    n_agents = 150

    def C_desc(self, s):
        g = self.G // s
        return f"count agents and add up their headings in a {g}x{g} grid"

    def reset(self):
        self.model = Vicsek(n_agents=self.n_agents, noise=float(self.param))
        self.rng = np.random.default_rng(self.seed)
        self.s = self.model.initial(self.rng)
        self.pol = deque(maxlen=300)
        self._record()

    def _record(self):
        self.pol.append(float(polarization(self.s)))

    def advance(self):
        for _ in range(2):
            self.s = self.model.step(self.s, self.rng)
        self._record()

    def perturb(self, rng):
        idx = rng.choice(len(self.s), size=len(self.s) // 3, replace=False)
        self.s[idx, 2] = rng.uniform(-np.pi, np.pi, len(idx))

    def _fields(self, states: np.ndarray) -> np.ndarray:
        """(..., N, 3) agent states -> (..., 3, G, G): density, sum cos, sum sin per cell."""
        lead = states.shape[:-2]
        flat = states.reshape(-1, *states.shape[-2:])
        L, G = self.model.box, self.G
        out = np.zeros((len(flat), 3, G * G))
        expected = self.model.n_agents / G**2
        for k, s in enumerate(flat):
            cell = (np.floor(s[:, 1] / L * G).astype(int) % G) * G + (
                np.floor(s[:, 0] / L * G).astype(int) % G
            )
            for c, w in enumerate((np.ones(len(s)), np.cos(s[:, 2]), np.sin(s[:, 2]))):
                out[k, c] = np.bincount(cell, weights=w, minlength=G * G)
        return (out / expected).reshape(*lead, 3, G, G)

    def _layer_fields(self):
        g = self.G // self._s
        f = self._fields(self.s).reshape(3, g, self._s, g, self._s).mean(axis=(2, 4))
        count = np.maximum(f[0], 1e-9)
        return f[0], f[1] / count, f[2] / count

    def setup_live(self, ax_micro, ax_layer, ax_macro, s):
        self._s = s
        st, L = self.s, self.model.box
        self._q_micro = ax_micro.quiver(
            st[:, 0], st[:, 1], np.cos(st[:, 2]), np.sin(st[:, 2]), st[:, 2], cmap="twilight",
            clim=(-np.pi, np.pi), angles="xy", scale_units="xy", scale=4, width=0.008)
        for ax in (ax_micro, ax_layer):
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        ax_micro.set_xlabel(f"{self.n_agents} agents (colour = heading)")

        g = self.G // s
        density, u, v = self._layer_fields()
        self._im_layer = ax_layer.imshow(density, origin="lower", extent=(0, L, 0, L),
                                         cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
        centres = (np.arange(g) + 0.5) * L / g
        X, Y = np.meshgrid(centres, centres)
        self._q_layer = ax_layer.quiver(X, Y, u, v, color="tab:red", angles="xy",
                                        scale_units="xy", scale=g / (0.8 * L), width=0.012)
        ax_layer.set_xlabel(f"{g}x{g} grid: shade = crowding, arrow = net motion")

        (self._line,) = ax_macro.plot([], [], color="tab:purple")
        ax_macro.set_xlim(0, self.pol.maxlen)
        ax_macro.set_ylim(0, 1)
        ax_macro.set_xlabel("time (last 300 frames)")
        ax_macro.set_ylabel("polarization |V| (1 = all aligned)")
        self.update_live()
        return [self._q_micro, self._im_layer, self._q_layer, self._line]

    def update_live(self):
        st = self.s
        self._q_micro.set_offsets(st[:, :2])
        self._q_micro.set_UVC(np.cos(st[:, 2]), np.sin(st[:, 2]), st[:, 2])
        density, u, v = self._layer_fields()
        self._im_layer.set_data(density)
        self._q_layer.set_UVC(u, v)
        p = np.array(self.pol)
        self._line.set_data(np.arange(len(p)), p)

    def ladder_field(self):
        runs = []
        for seed in range(3):
            traj = self.model.simulate(1500, 300, seed=seed)[::3]
            runs.append(self._fields(traj))
        return np.stack(runs)

    test_captions = (
        "AUTONOMY. Predict the flock's next change in polarization from polarization alone "
        "(blue) or also from every agent's heading (orange). If orange is no better, the "
        "flock-level variable is closed.",
        "DETERMINISM. Each dot is a moment of the flock. If the same polarization always leads "
        "to the same change, the dots form a thin curve; a wide cloud means the future depends "
        "on hidden details or on chance.",
        "ROBUSTNESS. Scramble the headings of 10% of the agents (dashed) and compare with the "
        "original (solid), using the same random noise. Lines that stay together mean the "
        "flock pattern survives changes to individuals.",
        "CAUSAL EMERGENCE (Psi). Does the flock's mean velocity predict its own future better "
        "than all agents individually? Psi > 0 means yes. Aligned agents share information, "
        "so the corrected Psi adds that redundancy back.",
    )

    def compute_tests(self):
        model = Vicsek(n_agents=100, noise=float(self.param))
        traj = model.simulate(4000, 500, seed=5)
        step = 10
        p = polarization(traj[::step])[:, None]
        micro = headings(traj[::step][:-1]).reshape(len(p) - 1, -1)
        groups = np.arange(len(p) - 1) // 25
        pc = predictive_closure(p[:-1], p[1:], micro, groups=groups)
        det = macro_determinism(p[:-1], p[1:], groups=groups)

        rng = np.random.default_rng(4)
        pairs = []
        for i in rng.choice(len(traj), 5, replace=False):
            a, b = traj[i].copy(), traj[i].copy()
            idx = rng.choice(len(b), size=10, replace=False)
            b[idx, 2] = rng.uniform(-np.pi, np.pi, 10)
            seed = int(rng.integers(2**31))
            ra, rb = np.random.default_rng(seed), np.random.default_rng(seed)
            a_line, b_line = [], []
            for _ in range(40):
                for _ in range(step):
                    a, b = model.step(a, ra), model.step(b, rb)
                a_line.append(polarization(a))
                b_line.append(polarization(b))
            pairs.append((np.array(a_line), np.array(b_line)))
        finals = np.array([a[-1] for a, _ in pairs])
        div = np.mean([(a[-1] - b[-1]) ** 2 for a, b in pairs])
        rob = 1 - div / (2 * finals.var()) if finals.var() > 0 else 1.0

        V = mean_velocity(traj)
        psi = emergence_psi(V[:-1], V[1:], headings(traj[:-1]))
        return {"pc": pc, "det": det, "z": p[:-1], "dz": p[1:] - p[:-1],
                "groups": groups % 10, "pairs": pairs, "rob": rob, "psi": psi,
                "zlabel": "polarization", "zdim": 0, "plabel": "polarization"}

    def draw_tests(self, axes):
        draw_continuous_tests(axes, self.tests())


def draw_continuous_tests(axes, t: dict):
    pc, d = t["pc"], t["zdim"]
    ax = axes[0]
    y = t["pc"]["target"][:, d]
    lim = np.percentile(np.abs(y), 99) * 1.2 or 1.0
    ax.scatter(y, pc["pred_macro"][:, d], s=6, alpha=0.5, color="tab:blue",
               label=f"macro only  R²={pc['r2_macro']:.2f}")
    ax.scatter(y, pc["pred_full"][:, d], s=6, alpha=0.5, color="tab:orange",
               label=f"macro + micro  R²={pc['r2_full']:.2f}")
    ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"actual change in {t['zlabel']}")
    ax.set_ylabel("predicted change")
    ax.legend(fontsize=8, loc="upper left")
    auto = pc["autonomy"]
    ax.set_title("Autonomy = " + ("n/a (unpredictable)" if np.isnan(auto) else f"{auto:.2f}"),
                 fontsize=9)

    ax = axes[1]
    ax.scatter(t["z"][:, d], t["dz"][:, d], c=t["groups"], cmap="tab10", s=6, alpha=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(f"{t['zlabel']} now")
    ax.set_ylabel(f"change in {t['zlabel']} next step")
    ax.set_title(f"Determinism = {t['det']:.2f}", fontsize=9)

    ax = axes[2]
    for k, (a, b) in enumerate(t["pairs"]):
        color = f"C{k}"
        ax.plot(a, color=color, lw=1.5)
        ax.plot(b, color=color, lw=1.5, ls="--")
    ax.set_xlabel("macro time after perturbation")
    ax.set_ylabel(t["plabel"])
    ax.set_title(f"Robustness = {t['rob']:.2f}", fontsize=9)

    ax = axes[3]
    psi = t["psi"]
    keys = ["I(V_t;V_t+1)", "max_I(X_j;V_t+1)", "sum_I(X_j;V_t+1)", "psi", "psi_corrected"]
    labels = ["macro→macro", "best single\npart→macro", "all parts\n(summed)", "Psi", "Psi\ncorrected"]
    vals = [psi[k] for k in keys]
    colors = ["tab:purple", "tab:blue", "tab:blue", "tab:red", "tab:red"]
    bars = ax.bar(labels, vals, color=colors)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_yscale("symlog", linthresh=1.0)
    lo, hi = min(vals + [0]), max(vals + [0])
    ax.set_ylim(-max(abs(lo) * 3, 2), max(hi * 3, 2))
    ax.set_ylabel("bits (log scale beyond ±1)")
    verdict = "emergent" if psi["psi_corrected"] > 0 else "not emergent"
    ax.set_title(f"Causal emergence: corrected Psi = {psi['psi_corrected']:.2f} ({verdict})",
                 fontsize=9)
    ax.tick_params(axis="x", labelsize=7)


ALL_VIEWS = (CAView, ChemView, FlockView)
