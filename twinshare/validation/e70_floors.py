"""e70's memory-gap floors with the lab's shifts (offsets in the middle half) and with whole-circle offsets.

Run from the lab's root (needs the lab's data):

    PYTHONPATH=. uv run python twinshare/validation/e70_floors.py

Drawing every member's offset from the middle half moves all members by about half the record, which keeps slow
regimes aligned and inflates the floor on long-memory series (FINDINGS, e70 revision).
"""

from __future__ import annotations

import numpy as np

from experiments import e70_screen as e70
from experiments import finance_data as fd
from experiments import memgap
from experiments.e67_memory_gap import HAR_WINDOWS

MIDDLE = memgap.shift


def whole_circle(segments, rng):
    out = []
    for s in segments:
        t, n = s.shape[:2]
        m = max(t // 20, 1)
        idx = (np.arange(t)[:, None] + rng.integers(m, t - m + 1, n)[None]) % t
        out.append(s[idx, np.arange(n)[None]])
    return out


def markets():
    sets = {}
    for name, (start, end) in {"industries 1972-2026": (19720101, 99999999),
                               "industries 1936-1971": (19360101, 19711231)}.items():
        _, r, _ = fd.industries("Average Value Weighted Returns", start, end)
        sets[name] = [np.log(np.abs(r) + 0.05)]
    _, x, _ = fd.stocks()
    sets["S&P 500 stocks 2013-2018"] = [np.log(x + 0.05)]
    return {k: (lambda v=v: memgap.screen(v, HAR_WINDOWS)) for k, v in sets.items()}


def main():
    rows = {}
    for label, fn in (("middle half (e70)", MIDDLE), ("whole circle", whole_circle)):
        memgap.shift = fn
        for name, run in markets().items():
            rows.setdefault(name, {})[label] = run()
        for r in e70.pigeons():
            rows.setdefault(r["name"], {})[label] = r
    memgap.shift = MIDDLE
    for name, by in rows.items():
        a, b = by["middle half (e70)"], by["whole circle"]
        print(f"{name:26s} S_pred {a['S_pred']:+.4f}   floor e70 {a['virtual']:+.4f}   floor whole circle "
              f"{b['virtual']:+.4f}")


if __name__ == "__main__":
    main()
