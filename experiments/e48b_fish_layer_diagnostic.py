"""Experiment 48b (exploratory, after e48): why did e48's model-based checks fail,
and what is the interaction-made block of real schools?

1. Group forecasts at 1 s and 3 s for recordings 0 and 3, from e44's k6-AC model
   with (full) and without (twin) social terms: per-variable interaction share
   of the group block, and the school's polarization and spread 3 s ahead.
2. From e48's raw results: the size of the top block of the 1 s spectrum (the
   largest jump), where its weight lies, and how much of its heading weight is
   relative to the school's mean heading.

    uv run python -m experiments.e48b_fish_layer_diagnostic
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import experiments.e44_elimination_speed_heading as m
import experiments.e48_fish_layers as e
from experiments.e31_real_fish import kinematics
from experiments.fish_data import load


def forecasts(rec):
    p, th, sp = kinematics(load(8, rec))
    T = len(th)
    cut = int(0.7 * T)
    fits, _, _, _, (l1, l2, _, dvp) = m.fit_all(p, th, sp, cut, [e.RULE])
    fit = fits[e.RULE]
    psi, w = e.dictionary(p, th, sp), e.group_block(8)
    idx = np.sort(np.random.default_rng(1).choice(np.arange(cut + 5, T - 31), 400, replace=False))
    out = {}
    for name, gain in (("full", 1.0), ("twin", 0.0)):
        r = np.random.default_rng(2)
        st = tuple(np.repeat(a[idx][:, None], 32, 1) for a in (p, th, sp, l1, l2, dvp))
        for s in range(1, 31):
            st = m.model_step(st, fit, e.RULE, r, gain)
            if s in (10, 30):
                out[(name, s)] = (e.dictionary(st[0], st[1], st[2]) @ w).mean(1)
        out[(name, "pol")] = float(np.abs(np.exp(1j * st[1]).mean(-1)).mean())
        out[(name, "spread")] = float(np.linalg.norm(st[0] - st[0].mean(-2, keepdims=True), axis=-1).mean())
    lines = []
    for s in (10, 30):
        truth = psi[idx + s] @ w
        ef = ((truth - out[("full", s)]) ** 2).mean(0)
        et = ((truth - out[("twin", s)]) ** 2).mean(0)
        lines.append(f"  #{rec} {s / 10:.0f} s: S per group variable (x, y, cos, sin, speed) "
                     + " ".join(f"{x:+.3f}" for x in 1 - ef / et))
    pol = float(np.abs(np.exp(1j * th[idx + 30]).mean(-1)).mean())
    spread = float(np.linalg.norm(p[idx + 30] - p[idx + 30].mean(-2, keepdims=True), axis=-1).mean())
    lines.append(f"  #{rec} 3 s ahead: polarization observed {pol:.2f}, full {out[('full', 'pol')]:.2f}, "
                 f"twin {out[('twin', 'pol')]:.2f}; spread observed {spread:.1f}, full "
                 f"{out[('full', 'spread')]:.1f}, twin {out[('twin', 'spread')]:.1f} cm")
    return lines


def top_block(raw):
    lines = []
    u = np.zeros((16, 2))
    u[:8, 0] = u[8:, 1] = 1 / np.sqrt(8)
    for r in sorted([r for r in raw["fish"] if r["kind"] == "real"], key=lambda r: r["rec"]):
        s, w = np.array(r["S_test"]), np.array(r["w"])
        ntop = 40 - (int(np.argmax(np.diff(s[::-1]))) + 1)
        top = w[:, :ntop] / np.linalg.norm(w[:, :ntop], axis=0)
        share = [float((top[b * 8:(b + 1) * 8] ** 2).sum() / ntop) for b in range(5)]
        h = top[16:32]
        rel = 1 - ((u.T @ h) ** 2).sum() / (h ** 2).sum()
        lines.append(f"  #{r['rec']}: top block {ntop} variables (S {s[:ntop].min():.2f}-{s[:ntop].max():.2f}, "
                     f"next {s[ntop]:.2f}); weight x,y,cos,sin,speed "
                     + " ".join(f"{x:.2f}" for x in share) + f"; heading weight relative to mean heading {rel:.3f}")
    return lines


def main(out_dir: Path | None = None) -> str:
    out_dir = out_dir or Path("results")
    raw = json.loads((out_dir / "e48_fish_layers_raw.json").read_text())
    lines = ["Experiment 48b (exploratory): e48 diagnostics", "", "1. Group forecasts, full vs twin"]
    for rec in (0, 3):
        lines += forecasts(rec)
    lines += ["", "2. Top block of the 1 s spectrum in real schools"] + top_block(raw)
    text = "\n".join(lines)
    (out_dir / "e48b_fish_layer_diagnostic.txt").write_text(text)
    print(text)
    return text


if __name__ == "__main__":
    main()
