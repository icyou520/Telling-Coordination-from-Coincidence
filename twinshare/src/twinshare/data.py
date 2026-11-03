"""Panels: many members measured on one shared clock, plus the recorded causes they all see.

A panel is read from CSV in one of two layouts:

  wide   one row per time step: a time column, one column per member, and optional covariate columns;
  long   one row per (time, member): a time column, a member column and a value column.

Covariates (weather, promotions, prices, holidays...) can sit in the wide file or in a separate CSV keyed by the same
time column. Everything done to the data (members dropped, gaps filled, transforms) is recorded in Panel.notes so the
report can show it.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MISSING = {"", "na", "nan", "null", "none", "-"}


@dataclass
class Panel:
    times: list[str]
    values: np.ndarray                     # (T, N) member variable, after the transform
    members: list[str]
    covariates: np.ndarray                 # (T, p) recorded causes shared by all members (may have p = 0)
    covariate_names: list[str]
    notes: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)   # file -> sha256

    @property
    def shape(self):
        return self.values.shape


def _num(s: str) -> float:
    s = s.strip()
    if s.lower() in MISSING:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return float(s.replace(",", ""))   # thousands separators, "1,234.5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return list(r.fieldnames or []), rows


def parse_date(s: str) -> dt.date | None:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _ordered(times: list[str]) -> list[str]:
    """Time stamps in time order: as dates if they all parse, else as numbers, else in the file's order."""
    dates = [parse_date(t) for t in times]
    if all(dates):
        return [t for _, t in sorted(zip(dates, times))]
    try:
        return [t for _, t in sorted((float(t), t) for t in times)]
    except ValueError:
        return times


def calendar(times: list[str], kind: str) -> tuple[np.ndarray, list[str], str | None]:
    """Calendar dummies for recorded seasonality. kind: "none", "weekday" or "month" (or "weekday+month")."""
    if kind == "none":
        return np.zeros((len(times), 0)), [], None
    dates = [parse_date(t) for t in times]
    if any(d is None for d in dates):
        return np.zeros((len(times), 0)), [], f"calendar '{kind}' skipped: the time column is not a date"
    cols, names = [], []
    parts = kind.split("+")
    for part, key, labels in (("weekday", lambda d: d.weekday(), "Mon Tue Wed Thu Fri Sat Sun".split()),
                              ("month", lambda d: d.month - 1, "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())):
        if part not in parts:
            continue
        k = np.array([key(d) for d in dates])
        for v in np.unique(k)[1:]:
            cols.append((k == v).astype(float))
            names.append(f"{part}={labels[v]}")
    unknown = set(parts) - {"weekday", "month"}
    if unknown:
        raise ValueError(f"unknown calendar part(s) {sorted(unknown)}; use none, weekday, month or weekday+month")
    return (np.column_stack(cols) if cols else np.zeros((len(times), 0))), names, None


def transform(x: np.ndarray, kind: str, offset: float) -> np.ndarray:
    if kind == "none":
        return x
    if kind == "log":
        if (x + offset <= 0).any():
            raise ValueError("transform 'log' needs x + offset > 0 everywhere; use 'logabs' or a larger offset")
        return np.log(x + offset)
    if kind == "logabs":
        return np.log(np.abs(x) + offset)
    raise ValueError(f"unknown transform {kind!r}; use none, log or logabs")


def _fill(x: np.ndarray, max_missing: float, members: list[str], notes: list[str]):
    """Drop members missing more than max_missing of steps; linearly interpolate the rest (ends held flat)."""
    frac = np.isnan(x).mean(0)
    keep = frac <= max_missing
    if (~keep).any():
        dropped = [m for m, k in zip(members, keep) if not k]
        notes.append(f"dropped {len(dropped)} member(s) missing more than {max_missing:.0%} of steps: "
                     + ", ".join(dropped[:10]) + (" ..." if len(dropped) > 10 else ""))
    x, members = x[:, keep], [m for m, k in zip(members, keep) if k]
    gaps = int(np.isnan(x).sum())
    if gaps:
        t = np.arange(len(x))
        for i in range(x.shape[1]):
            bad = np.isnan(x[:, i])
            if bad.any():
                x[bad, i] = np.interp(t[bad], t[~bad], x[~bad, i])
        notes.append(f"filled {gaps} missing value(s) ({gaps / x.size:.2%} of the panel) by linear interpolation")
    return x, members


def load(study: dict, root: Path = Path(".")) -> Panel:
    """Read the panel described by a study's [data] and [model] tables."""
    d, m = study["data"], study.get("model", {})
    path = root / d["file"]
    fields, rows = _read(path)
    tcol = d.get("time", fields[0])
    covs = list(d.get("covariates", []))
    sources = {str(d["file"]): sha256(path)}
    notes: list[str] = []

    if d.get("format", "wide") == "long":
        mcol, vcol = d["member"], d["value"]
        times = _ordered(list(dict.fromkeys(r[tcol] for r in rows)))
        members = sorted({r[mcol] for r in rows})
        ti, mi = {t: k for k, t in enumerate(times)}, {s: k for k, s in enumerate(members)}
        x = np.full((len(times), len(members)), np.nan)
        for r in rows:
            x[ti[r[tcol]], mi[r[mcol]]] = _num(r[vcol])
        cov_rows = {}
    else:
        members = list(d.get("members") or [c for c in fields if c != tcol and c not in covs])
        missing = [c for c in members + covs if c not in fields]
        if missing:
            raise ValueError(f"columns not in {path.name}: {missing}")
        times = [r[tcol] for r in rows]
        x = np.array([[_num(r[c]) for c in members] for r in rows])
        cov_rows = {r[tcol]: r for r in rows}
    if len(set(times)) != len(times):
        raise ValueError(f"time column '{tcol}' has repeated values; one row per time step is required")

    if d.get("covariates_file"):
        cpath = root / d["covariates_file"]
        cfields, crow_list = _read(cpath)
        sources[str(d["covariates_file"])] = sha256(cpath)
        cov_rows = {r[d.get("covariates_time", tcol)]: r for r in crow_list}
        covs = covs or [c for c in cfields if c != d.get("covariates_time", tcol)]
    if covs:
        absent = [t for t in times if t not in cov_rows]
        if absent:
            raise ValueError(f"covariates missing for {len(absent)} time step(s), first {absent[0]}")
        z = np.array([[_num(cov_rows[t][c]) for c in covs] for t in times])
        if np.isnan(z).any():
            raise ValueError("covariates contain missing values; fill them or drop those time steps first")
    else:
        z = np.zeros((len(times), 0))

    x, members = _fill(x, float(d.get("max_missing", 0.1)), members, notes)
    kind, offset = m.get("transform", "none"), float(m.get("offset", 0.0))
    x = transform(x, kind, offset)
    if kind != "none":
        notes.append(f"transform: {kind}(x{' + ' + str(offset) if offset else ''})")
    cal, cal_names, warn = calendar(times, m.get("calendar", "none"))
    if warn:
        notes.append(warn)
    if cal_names:
        notes.append(f"calendar: {len(cal_names)} {m.get('calendar')} dummies added to the recorded causes")
    const = [k for k in range(x.shape[1]) if np.std(x[:, k]) == 0]
    if const:
        notes.append(f"dropped {len(const)} constant member(s)")
        keep = np.setdiff1d(np.arange(x.shape[1]), const)
        x, members = x[:, keep], [members[k] for k in keep]
    if x.shape[1] < 2:
        raise ValueError("need at least two members with data")
    return Panel(times, x, members, np.column_stack([z, cal]), covs + cal_names, notes, sources)


def from_arrays(values: np.ndarray, covariates: np.ndarray | None = None, members=None, times=None) -> Panel:
    """A panel from arrays (for use from Python): values (T, N), covariates (T, p)."""
    values = np.asarray(values, float)
    t, n = values.shape
    z = np.zeros((t, 0)) if covariates is None else np.asarray(covariates, float).reshape(t, -1)
    return Panel(list(times) if times is not None else [str(k) for k in range(t)], values,
                 list(members) if members is not None else [f"m{k}" for k in range(n)], z,
                 [f"z{k}" for k in range(z.shape[1])])
