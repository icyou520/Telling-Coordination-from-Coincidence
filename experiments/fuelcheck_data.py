"""FuelCheck (New South Wales): every station-level price change since August 2016, and the Australian Institute of
Petroleum's daily terminal gate prices (TGP) by city since 2004.

    uv run python -m experiments.fuelcheck_data      # downloads into data/fuelcheck/ (about 50 MB streamed)

FuelCheck's monthly price-history workbooks (data.nsw.gov.au dataset "fuel-check", CC BY 4.0) have one row per price
change: station name, address, suburb, postcode, brand, fuel code, time of the change (local time) and price (cpl).
Only regular unleaded (U91) is kept, for August 2016 to December 2017 (the period of Byrne, de Roos, Lewis, Marx and
Wu), cached as data/fuelcheck/retail/YYYY-MM.csv.gz. The AIP workbook (AIP_TGP_Data_*.xlsx, aip.com.au) gives the
average ULP TGP for each capital city on business days; the Sydney column is cached as data/fuelcheck/tgp_sydney.csv.
Workbooks are read with the standard library (zip + XML), so no spreadsheet package is needed.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import re
import subprocess
import sys
import time
import zipfile
import io
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("data/fuelcheck")
CKAN = "https://data.nsw.gov.au/data/api/3/action/package_show?id=fuel-check"
AIP_PAGE = "https://www.aip.com.au/historical-ulp-and-diesel-tgp-data"
MONTHS = [f"{y}-{m:02d}" for y in (2016, 2017) for m in range(1, 13) if (y, m) >= (2016, 8)]
RETAIL_COLS = ["time", "station", "address", "suburb", "postcode", "brand", "price"]
MONTH_NAMES = {m: k for k, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                           "nov", "dec"], 1)}
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _get(url, tries=4):
    for k in range(tries):
        out = subprocess.run(["curl", "-sfL", "-m", "300", "-A", "Mozilla/5.0", url], capture_output=True)
        if out.returncode == 0:
            return out.stdout
        if k == tries - 1:
            raise RuntimeError(f"curl {out.returncode}: {url}")
        time.sleep(2 ** k)


def xlsx_rows(raw, sheet=1):
    """Rows of one worksheet as lists of strings (numbers as written in the file, dates as Excel serials)."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.parse(z.open("xl/sharedStrings.xml")).getroot().iter(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    for _, row in ET.iterparse(z.open(f"xl/worksheets/sheet{sheet}.xml")):
        if row.tag != NS + "row":
            continue
        out = {}
        for c in row.iter(NS + "c"):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            k = 0
            for ch in col:
                k = 26 * k + ord(ch) - 64
            v = c.find(NS + "v")
            if c.get("t") == "s":
                out[k - 1] = shared[int(v.text)]
            elif c.get("t") == "inlineStr":
                out[k - 1] = "".join(t.text or "" for t in c.iter(NS + "t"))
            else:
                out[k - 1] = v.text if v is not None else ""
        row.clear()
        yield [out.get(i, "") for i in range(max(out) + 1)] if out else []


def _when(s):
    """Excel serial or text timestamp -> ISO 'YYYY-MM-DD HH:MM:SS'."""
    try:
        t = dt.datetime(1899, 12, 30) + dt.timedelta(days=float(s))
        return (t + dt.timedelta(microseconds=500_000)).replace(microsecond=0).isoformat(" ")
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %I:%M:%S %p"):
            try:
                return dt.datetime.strptime(s.strip(), fmt).isoformat(" ")
            except ValueError:
                pass
    raise ValueError(f"unreadable time {s!r}")


def _month_of(name):
    """'Fuelcheck Price History September 2016' -> '2016-09'."""
    words = name.lower().replace(".xlsx", "").split()
    year = next(w for w in words if re.fullmatch(r"20\d\d", w))
    month = next(MONTH_NAMES[w[:3]] for w in words if w[:3] in MONTH_NAMES)
    return f"{year}-{month:02d}"


def _write(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    tmp.rename(path)


def download():
    resources = json.loads(_get(CKAN))["result"]["resources"]
    urls = {}
    for r in resources:
        if r.get("url", "").lower().endswith(".xlsx"):
            try:
                urls[_month_of(r["name"])] = r["url"]
            except StopIteration:
                pass
    for ym in MONTHS:
        path = ROOT / "retail" / f"{ym}.csv.gz"
        if path.exists():
            continue
        rows = xlsx_rows(_get(urls[ym]))
        for _ in range(10):                          # some months have a title row above the header
            header = [h.strip() for h in next(rows)]
            if header[:1] == ["ServiceStationName"]:
                break
        assert header[0] == "ServiceStationName" and header[7] == "Price", header
        keep = [[_when(r[6]), r[0].strip(), r[1].strip(), r[2].strip(), r[3].strip(), r[4].strip(), r[7]]
                for r in rows if len(r) > 7 and r[5].strip() == "U91" and r[7]]
        _write(path, RETAIL_COLS, keep)
        print(f"{path}  {len(keep):,} U91 price changes", flush=True)
    path = ROOT / "tgp_sydney.csv"
    if not path.exists():
        page = _get(AIP_PAGE).decode("utf-8", errors="replace")
        url = sorted(set(re.findall(r"https?://[^\"' ]*AIP_TGP_Data_[^\"' ]*\.xlsx", page)))[-1]
        sheets = zipfile.ZipFile(io.BytesIO(raw := _get(url))).read("xl/workbook.xml").decode()
        names = re.findall(r'<sheet [^>]*name="([^"]+)"', sheets)
        rows = xlsx_rows(raw, names.index("Petrol TGP") + 1)
        header = next(rows)
        col = [h.strip() for h in header].index("Sydney")
        out = [[_when(r[0])[:10], r[col]] for r in rows if len(r) > col and r[0] and r[col]
               and re.fullmatch(r"[\d.]+", r[0])]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([["date", "tgp"]] + out)
        print(f"{path}  {len(out):,} days ({url.rsplit('/', 1)[-1]})", flush=True)


def retail(year_month):
    """U91 price changes for one month ('YYYY-MM') as a list of dicts."""
    with gzip.open(ROOT / "retail" / f"{year_month}.csv.gz", "rt") as f:
        return list(csv.DictReader(f))


def tgp_sydney():
    with open(ROOT / "tgp_sydney.csv") as f:
        return {r["date"]: float(r["tgp"]) for r in csv.DictReader(f)}


if __name__ == "__main__":
    download()
    sys.exit(0)
