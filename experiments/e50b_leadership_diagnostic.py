"""Experiment 50b (exploratory, after e50): does the front-fish leadership signal
survive the school's own turning history, out of sample?

e50's closure test (20 configuration features and their squares) found no
gain for the school's motion, yet the single leadership feature L was highly
significant. Here, for each real 8-fish school: the change of the mean heading
over 1 s regressed on [1, the last 5 turning rates, polarization, spread] with
and without L; Newey-West t of L, in-sample R^2, and the held-out gain
1 - MSE(with L) / MSE(without), fitted on the first 70%.

    uv run python -m experiments.e50b_leadership_diagnostic
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import experiments.e50_school_configuration as e
from experiments.e31_real_fish import kinematics
from experiments.e49_school_layer import school_state
from experiments.fish_data import load


def main(out_dir: Path | None = None) -> str:
    out_dir = out_dir or Path("results")
    lines = ["Experiment 50b (exploratory): leadership signal vs the school's own turning history", ""]
    for rec in range(6):
        p, th, sp = kinematics(load(8, rec))
        T = len(th)
        _, lead, big_th = e.configuration(p, th, sp)
        z = school_state(p, th, sp)
        rows = np.arange(10, T - 10)
        d = e._wrap(big_th[rows + 10] - big_th[rows])
        rates = [e._wrap(big_th[rows - j] - big_th[rows - j - 1]) for j in range(5)]
        xb = np.column_stack([np.ones(len(rows))] + rates + [z[rows, 5], z[rows, 6]])
        xl = np.column_stack([xb, lead[rows]])
        b, se = e.newey_west(xl, d, e.NW_LAGS)
        r2b = 1 - ((d - xb @ np.linalg.lstsq(xb, d, rcond=None)[0]) ** 2).mean() / d.var()
        r2l = 1 - ((d - xl @ b) ** 2).mean() / d.var()
        tr = rows < int(0.7 * T)
        eb = d[~tr] - xb[~tr] @ np.linalg.lstsq(xb[tr], d[tr], rcond=None)[0]
        el = d[~tr] - xl[~tr] @ np.linalg.lstsq(xl[tr], d[tr], rcond=None)[0]
        lines.append(f"  8 fish #{rec}: L t {b[-1] / se[-1]:5.1f}; R^2 {r2b:.3f} -> {r2l:.3f}; "
                     f"held-out gain {1 - (el ** 2).mean() / (eb ** 2).mean():+.3f}")
    text = "\n".join(lines)
    (out_dir / "e50b_leadership_diagnostic.txt").write_text(text)
    print(text)
    return text


if __name__ == "__main__":
    main()
