"""Loaders for daily market data in data/finance/.

  49_Industry_Portfolios_Daily.csv  Kenneth French data library, value-weighted daily returns (%), 49 industries.
  all_stocks_5yr.csv                S&P 500 constituents, daily OHLC, Feb 2013 - Feb 2018 (plotly/datasets).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path("data/finance")
START = 19720101


@lru_cache(maxsize=4)
def industry_section(title="Average Value Weighted Returns"):
    """All rows of one section of the 49-industry file: (dates, returns (days, 49) with NaN for missing, names)."""
    return _section((DATA / "49_Industry_Portfolios_Daily.csv").read_text().splitlines(), title)


REGIONS = ("Japan", "Europe", "Asia_Pacific_ex_Japan")


def regional(region, title="Average Value Weighted Returns"):
    """Daily returns (%) of a region's 25 size x book-to-market portfolios (French library, 1990 on), keeping the
    portfolios with no missing day: (dates, returns (days, portfolios), names)."""
    import zipfile
    with zipfile.ZipFile(DATA / f"{region}_25_Daily.zip") as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    dates, x, names = _section(lines, title)
    keep = np.isfinite(x).all(0)
    return dates, x[:, keep], [n for n, k in zip(names, keep) if k]


def _section(lines, title):
    head = next(i for i, l in enumerate(lines) if title in l)
    names = [n.strip() for n in lines[head + 1].split(",")[1:]]
    rows = []
    for l in lines[head + 2:]:
        parts = l.split(",")
        if len(parts) != len(names) + 1 or not parts[0].strip().isdigit():
            break
        rows.append([float(p) for p in parts])
    a = np.array(rows)
    x = a[:, 1:]
    x[x <= -99] = np.nan
    return a[:, 0].astype(int), x, names


def industries(title="Average Value Weighted Returns", start=START, end=99999999):
    """Daily returns (%) between start and end of the industries with no missing day in that span."""
    dates, x, names = industry_section(title)
    sel = (dates >= start) & (dates <= end)
    keep = np.isfinite(x[sel]).all(0)
    return dates[sel], x[sel][:, keep], [n for n, k in zip(names, keep) if k]


@lru_cache(maxsize=1)
def stocks():
    """Daily log range log(high/low) (%) of stocks with every trading day: (dates, ranges (days, stocks), names)."""
    import csv
    by = {}
    with open(DATA / "all_stocks_5yr.csv") as f:
        for r in csv.DictReader(f):
            try:
                hi, lo = float(r["high"]), float(r["low"])
            except ValueError:
                continue
            if lo > 0 and hi >= lo:
                by.setdefault(r["Name"], {})[r["date"]] = 100 * np.log(hi / lo)
    dates = sorted({d for v in by.values() for d in v})
    names = sorted(n for n, v in by.items() if len(v) == len(dates))
    x = np.array([[by[n][d] for n in names] for d in dates])
    return np.array([int(d.replace("-", "")) for d in dates]), x, names
