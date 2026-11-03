"""twinshare against the Emergence Lab's published numbers (needs the lab repository and its data).

Run from the lab's root:

    PYTHONPATH=twinshare/src uv run python twinshare/validation/lab_replication.py

1. With the lab's settings (250-day blocks, no purge, penalty 1, causes at t only) twinshare's share equals e76's
   leak-free share on e66's simulated markets and on the US industries (FINDINGS e76), to rounding error.
2. With twinshare's defaults the same data give nearly the same shares (the defaults differ in safeguards, not
   in what is measured).
"""

from __future__ import annotations

import numpy as np

from experiments import e66_market_volatility as e66
from experiments import e76_leak_audit as e76
from experiments import finance_data as fd
from twinshare import engine as E

LAB = E.Design(windows=(1, 5, 22), block=250, purge=False, lam=1.0, lagged_causes=False)
DEFAULT = E.Design(windows=(1, 5, 22))


def compare(name, x, dow):
    lab = e76.share_series(x, dow)
    s_lab = 1 - lab[0].sum() / lab[1].sum()
    z = np.eye(5)[dow][:, 1:]
    s_ts = E.share(x, z, LAB).S
    s_def = E.share(x, z, DEFAULT).S
    gap = E.memory_gap(x, z, DEFAULT)["S_pred"]
    print(f"{name:34s} lab {s_lab:+.5f}  twinshare (lab settings) {s_ts:+.5f}  |diff| {abs(s_lab - s_ts):.1e}  "
          f"twinshare defaults {s_def:+.5f}  memory gap {gap:+.5f}")
    return abs(s_lab - s_ts)


def main():
    worst = 0.0
    for cond in ("independent", "spillover", "factor"):
        x = np.log(e66.simulate(cond, 20) + 0.05)
        worst = max(worst, compare(f"simulated market, {cond}", x, e66._dow(len(x))))
    for title, start, end, label in (("Average Equal Weighted Returns", 19720101, 99999999, "US industries, equal 1972-2026"),
                                     ("Average Value Weighted Returns", 19360101, 19711231, "US industries, value 1936-1971")):
        dates, r, _ = fd.industries(title, start, end)
        worst = max(worst, compare(label, np.log(np.abs(r) + 0.05), e66._dow_dates(dates)))
    print(f"\nlargest difference from the lab with its settings: {worst:.1e}")

    from twinshare import analyze, from_arrays
    dates, r, names = fd.industries("Average Equal Weighted Returns", 19720101, 99999999)
    p = from_arrays(np.log(np.abs(r) + 0.05), np.eye(5)[e66._dow_dates(dates)][:, 1:], names, map(str, dates))
    res = analyze(p, DEFAULT, seed=0)
    print(f"\nfull analysis, US industries 1972-2026 (equal-weighted log volatility): {res.verdict}")
    print(res.explanation)
    for c in res.checks:
        print(f"  {c.status:5s} {c.name}: {c.value}")


if __name__ == "__main__":
    main()
