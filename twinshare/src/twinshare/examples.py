"""Synthetic data sets with a known answer, for learning the tool and checking it.

retail: daily sales of a chain of stores. Every store sees the weather, chain-wide promotions and the weekday
pattern (recorded causes), and has its own persistent ups and downs. Optionally:
  coupling   stores respond to how the whole chain did yesterday (word of mouth, shared stock-outs): interaction;
  hidden     a slow regional economy that moves every store a little and is not recorded: a shared driver;
  leader     every other store follows store_01's previous day (a flagship that sets trends): interaction with a
             structure, which the closure and influence analysis should find.
With neither, the answer is "no coordination". Both produce a positive share; the share cannot tell them apart,
which is the point of declaring what is recorded.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np


def retail(seed=0, stores=50, days=1095, coupling=0.0, hidden=0.0, leader=0.0, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    d0 = dt.date.fromisoformat(start)
    dates = [d0 + dt.timedelta(days=k) for k in range(days)]
    doy = np.array([d.timetuple().tm_yday for d in dates])
    weekday = np.array([d.weekday() for d in dates])
    temp = 15 + 10 * np.sin(2 * np.pi * (doy - 100) / 365) + np.convolve(rng.normal(0, 2, days + 6),
                                                                          np.ones(7) / 7, "valid")
    promo = (rng.random(days) < 0.08).astype(float)
    week = np.array([0.0, -0.05, -0.03, 0.0, 0.1, 0.3, 0.2])[weekday]
    level = rng.normal(np.log(8.0), 0.3, stores)
    temp_eff = rng.normal(0.01, 0.003, stores)
    promo_eff = rng.normal(0.15, 0.04, stores)
    c = np.zeros(days)
    for t in range(1, days):
        c[t] = 0.98 * c[t - 1] + rng.normal(0, 0.02)
    u = np.zeros((days, stores))
    shock = rng.normal(0, 0.08, (days, stores))
    for t in range(1, days):
        u[t] = 0.6 * u[t - 1] + shock[t] + coupling * u[t - 1].mean() + hidden * c[t]
        u[t, 1:] += leader * u[t - 1, 0]
    log_sales = level + week[:, None] + temp_eff * temp[:, None] + promo_eff * promo[:, None] + u
    sales = np.round(np.exp(log_sales) * 100) / 100
    return dates, sales, temp, promo


def write_retail(path: Path, **kw) -> Path:
    dates, sales, temp, promo = retail(**kw)
    n = sales.shape[1]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [f"store_{k + 1:02d}" for k in range(n)] + ["temperature", "promotion"])
        for k, d in enumerate(dates):
            w.writerow([d.isoformat()] + [f"{v:.2f}" for v in sales[k]] + [f"{temp[k]:.1f}", int(promo[k])])
    return path
