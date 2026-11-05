"""FuelWatch (Western Australia): daily station-level retail prices since January 2001 and daily terminal gate prices.

    uv run python -m experiments.fuelwatch_data      # downloads into data/fuelwatch/ (about 2.4 GB streamed, ~60 MB kept)

FuelWatch requires every retailer to submit tomorrow's price for each station by 2 p.m.; the price then holds for 24
hours from 6 a.m. Monthly retail files (FuelWatchRetail-MM-YYYY.csv) have one row per station, product and day; yearly
wholesale files (FuelWatchWholesale-YYYY.csv) one row per terminal gate, product and day. Licence CC BY 4.0. The file
index comes from the public endpoints /api/report/monthly-retail-prices and /api/report/terminal-gate-prices.

Only regular unleaded (ULP) is kept. Raw files are not stored: each is reduced to its ULP rows and cached as
data/fuelwatch/retail/YYYY-MM.csv.gz and data/fuelwatch/wholesale/YYYY.csv.gz.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("data/fuelwatch")
API = "https://www.fuelwatch.wa.gov.au/api/report/"
RETAIL_COLS = ["date", "station", "brand", "price", "address", "location", "postcode", "area", "region"]
WHOLESALE_COLS = ["date", "terminal", "brand", "price", "location"]


def _get(url, tries=4):
    for k in range(tries):
        out = subprocess.run(["curl", "-sfL", "-m", "300", "-H", "Accept: application/json", url], capture_output=True)
        if out.returncode == 0:
            return out.stdout
        if k == tries - 1:
            raise RuntimeError(f"curl {out.returncode}: {url}")
        time.sleep(2 ** k)


def _index(kind):
    return json.loads(_get(API + kind))


def _iso(d):
    day, month, year = d.split("/")
    return f"{year}-{month}-{day}"


def _reduce(raw, keep):
    rows = csv.reader(io.StringIO(raw.decode("utf-8-sig", errors="replace")))
    header = [h.strip().upper() for h in next(rows)]
    out = []
    for r in rows:
        d = dict(zip(header, r))
        if d.get("PRODUCT_DESCRIPTION", d.get("PRODUCT", "")).strip() == "ULP":
            out.append(keep(d))
    return out


def _retail_row(r):
    return [_iso(r["PUBLISH_DATE"]), r["TRADING_NAME"].strip(), r["BRAND_DESCRIPTION"].strip(), r["PRODUCT_PRICE"],
            r["ADDRESS"].strip(), r["LOCATION"].strip(), r["POSTCODE"].strip(), r["AREA_DESCRIPTION"].strip(),
            r["REGION_DESCRIPTION"].strip()]


def _wholesale_row(r):
    return [_iso(r["PUBLISH_DATE"]), r["TERMINAL_GATE"].strip(), r["BRAND"].strip(), r["PRODUCT_PRICE"], r["LOCATION"].strip()]


def _write(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    tmp.rename(path)


def download():
    jobs = []
    for e in _index("monthly-retail-prices"):
        month, year = e["fileName"].removeprefix("FuelWatchRetail-").removesuffix(".csv").split("-")
        jobs.append((e["url"], ROOT / "retail" / f"{year}-{month}.csv.gz", RETAIL_COLS, _retail_row))
    for e in _index("terminal-gate-prices"):
        year = e["fileName"].removeprefix("FuelWatchWholesale-").removesuffix(".csv")
        jobs.append((e["url"], ROOT / "wholesale" / f"{year}.csv.gz", WHOLESALE_COLS, _wholesale_row))
    for url, path, cols, keep in sorted(jobs, key=lambda j: str(j[1])):
        if path.exists():
            continue
        rows = _reduce(_get(url), keep)
        _write(path, cols, rows)
        print(f"{path}  {len(rows):,} ULP rows", flush=True)


def retail(year_month):
    """ULP rows for one month ('YYYY-MM') as a list of dicts."""
    with gzip.open(ROOT / "retail" / f"{year_month}.csv.gz", "rt") as f:
        return list(csv.DictReader(f))


def wholesale(year):
    with gzip.open(ROOT / "wholesale" / f"{year}.csv.gz", "rt") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    download()
    sys.exit(0)
