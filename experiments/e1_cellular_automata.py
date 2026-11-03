"""Experiment 1: cellular automata, where every micro rule is known.

Part A asks a fixed question of all 256 rules: is global density an
autonomous macro variable? (exact closure gap by full enumeration)

Part B lets the computer *discover* emergent layers: it searches every
block projection for every rule and keeps those for which an exact macro
rule G exists, i.e. C(F(x)) = G(C(x)) holds for all configurations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emergence_lab import closure_gap_bits
from emergence_lab.systems import eca


def density_closure(n_cells: int = 12) -> dict[int, float]:
    x = eca.all_states(n_cells)
    z = x.sum(1)
    return {rule: closure_gap_bits(z, eca.step(x, rule).sum(1)) for rule in range(256)}


def discover(blocks=(2, 3)) -> list[eca.CoarseGraining]:
    found = []
    for rule in range(256):
        for b in blocks:
            found += eca.find_coarse_grainings(rule, b)
    return found


def plot_example(cg: eca.CoarseGraining, path: Path, n_cells: int = 240, n_macro: int = 60):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    x0 = rng.integers(0, 2, n_cells - n_cells % cg.block, dtype=np.uint8)
    micro = eca.run(x0, cg.rule, n_macro * cg.block)
    via_micro = cg.coarse(micro[:: cg.block])
    via_macro = eca.run(cg.coarse(x0), cg.macro_rule, n_macro)

    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    ax[0].imshow(micro, cmap="binary", aspect="auto", interpolation="nearest")
    ax[0].set_title(f"micro: rule {cg.rule}")
    ax[1].imshow(via_micro, cmap="binary", aspect="auto", interpolation="nearest")
    ax[1].set_title("path A: simulate micro, then coarse-grain")
    ax[2].imshow(via_macro, cmap="binary", aspect="auto", interpolation="nearest")
    ax[2].set_title(f"path B: emergent rule {cg.macro_rule} only")
    for a in ax:
        a.set_xlabel("space")
    ax[0].set_ylabel("time")
    fig.suptitle(f"{cg.describe()}   (paths identical: {np.array_equal(via_micro, via_macro)})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(out_dir: Path | None = None) -> str:
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say("== A. Is density an autonomous macro variable? (all 256 rules, 12-cell ring, exact) ==")
    gaps = density_closure()
    closed = sorted(r for r, g in gaps.items() if g < 1e-12)
    say(f"  density is exactly closed for {len(closed)} rules: {closed}")
    order = sorted(gaps, key=gaps.get, reverse=True)
    say("  largest gaps H(Z'|Z): " + ", ".join(f"rule {r}={gaps[r]:.2f}b" for r in order[:5]))
    say("  famous rules: " + ", ".join(f"{r}={gaps[r]:.2f}b" for r in (30, 90, 110, 184)))

    say("\n== B. Automatic discovery of exact coarse-grainings (block sizes 2 and 3) ==")
    found = discover()
    nontrivial = [c for c in found if not c.trivial]
    rules_with = sorted({c.rule for c in nontrivial})
    say(f"  {len(found)} exact coarse-grainings found; {len(nontrivial)} have a non-trivial "
        f"macro rule, spanning {len(rules_with)} micro rules")
    best = {}
    for c in nontrivial:
        best.setdefault((c.rule, c.macro_rule), c)
    say("  examples (micro rule -> emergent macro rule):")
    for (r, g), c in sorted(best.items())[:25]:
        say(f"    {c.describe()}")
    if len(best) > 25:
        say(f"    ... and {len(best) - 25} more distinct (micro, macro) pairs")

    say("\n  independent check on a 600-cell ring, 100 macro steps:")
    sample = [c for c in nontrivial if c.rule != c.macro_rule][:8]
    for c in sample:
        say(f"    {c.describe():<60} mismatches = {eca.verify_on_lattice(c, 600, 100)}")
    for famous in (30, 110):
        say(f"  rule {famous}: {sum(c.rule == famous for c in nontrivial)} non-trivial "
            "coarse-grainings at block size <= 3")

    if out_dir is not None:
        example = next(c for c in nontrivial if c.rule == 105)
        plot_example(example, out_dir / "e1_ca_coarse_graining.png")
        say(f"\n  figure: {out_dir / 'e1_ca_coarse_graining.png'}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
