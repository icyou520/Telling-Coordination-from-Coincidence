"""Experiment 0: the toy systems behind the lab's starting hypothesis.

* Two flipping bits with macro "do they differ?" (the Lean example).
* A 3-bit counter with several candidate macro variables, some closed, some not.
* XOR: information that exists only in the combination of parts.
"""

from __future__ import annotations

import itertools

import numpy as np

from emergence_lab import closure_gap_bits, compression_bits, exact_closure, synergy_report


def main(out_dir=None) -> str:
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say("== Flip pair: micro (a, b) -> (not a, not b); macro = a XOR b ==")
    states = list(itertools.product([False, True], repeat=2))
    res = exact_closure(states, lambda s: (not s[0], not s[1]), lambda s: s[0] != s[1])
    say(f"  {res}")

    say("\n== 3-bit counter: x -> x + 1 mod 8 ==")
    states = list(range(8))
    F = lambda x: (x + 1) % 8
    macros = {
        "parity (x mod 2)": lambda x: x % 2,
        "x mod 4": lambda x: x % 4,
        "high bit (x >= 4)": lambda x: int(x >= 4),
        "popcount (# of 1 bits)": lambda x: bin(x).count("1"),
    }
    xs = np.array(states)
    for name, C in macros.items():
        res = exact_closure(states, F, C)
        z = np.array([C(x) for x in states])
        zn = np.array([C(F(x)) for x in states])
        say(f"  {name:<24} gap={closure_gap_bits(z, zn):.3f} bits  "
            f"compression={compression_bits(xs, z):.2f}  {res}")
        for zval, wit in res.witnesses.items():
            ex = ", ".join(f"x={x} -> {zn_}" for zn_, x in wit.items())
            say(f"      counterexample at macro={zval}: {ex}")

    say("\n== XOR synergy ==")
    a, b = np.array(list(itertools.product([0, 1], repeat=2))).T
    rep = synergy_report(a, b, a ^ b)
    say("  " + "  ".join(f"{k}={v:.3f}" for k, v in rep.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
