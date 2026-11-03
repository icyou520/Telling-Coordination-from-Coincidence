"""Fresh stock markets for e82: markets and firms that e79 and e81 never touched.

    uv run --with pandas --with lxml python -m experiments.stock_data_fresh constituents   # Wikipedia, once
    uv run python -m experiments.stock_data_fresh prices                                   # Yahoo Finance, once

Constituents: today's members of 13 indices outside e79's eight, from English Wikipedia on the fetch date
(data/stocks_fresh/constituents.json), plus "US other": S&P 500 members that are not in the S&P 100 (e79's US panel).
Prices: Yahoo Finance daily adjusted closes from 2000 (data/stocks_fresh/prices/<ticker>.csv).
The panel rule is e79's (experiments/stock_data.py), unchanged: same dates, coverage, cap, at most 40 members (seeded
subset); a market is used if it has >= 15 members. Survivors only, as in e79.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path
import numpy as np

from experiments import stock_data as sd

DATA = Path("data/stocks_fresh")
W = "https://en.wikipedia.org/wiki/"

PAGES = {
    "Switzerland": ("SMI", W + "Swiss_Market_Index", "Ticker", lambda s: s + ".SW"),
    "Netherlands": ("AEX", W + "AEX_index", "Ticker", str),
    "Spain": ("IBEX 35", W + "IBEX_35", "Ticker", str),
    "Italy": ("FTSE MIB", W + "FTSE_MIB", "Ticker", str),
    "Sweden": ("OMX Stockholm 30", W + "OMX_Stockholm_30", "Ticker", str),
    "Finland": ("OMX Helsinki 25", W + "OMX_Helsinki_25", "Ticker", str),
    "Norway": ("OBX", W + "OBX_Index", "Ticker symbol", lambda s: s.split(":")[-1].strip() + ".OL"),
    "Belgium": ("BEL 20", W + "BEL20", "Ticker symbol", lambda s: s.split(":")[-1].strip() + ".BR"),
    "India": ("NIFTY 50", W + "NIFTY_50", "Symbol", lambda s: s + ".NS"),
    "Korea": ("KOSPI 200", W + "KOSPI_200", "Symbol", lambda s: str(s).zfill(6) + ".KS"),
    "Singapore": ("STI", W + "Straits_Times_Index", "Stock symbol", lambda s: s.split(":")[-1].strip() + ".SI"),
    "New Zealand": ("S&P/NZX 50", W + "S%26P%2FNZX_50_Index", "Ticker symbol", lambda s: s + ".NZ"),
    "Brazil": ("B3 listed", W + "List_of_companies_listed_on_B3", "Ticker", lambda s: s + ".SA"),
    "US other": ("S&P 500 not in S&P 100", W + "List_of_S%26P_500_companies", "Symbol", lambda s: s.replace(".", "-")),
}
MARKETS = tuple(PAGES)


def fetch_constituents():
    import io

    import pandas as pd
    sp100 = set(json.loads((sd.DATA / "constituents.json").read_text())["US"]["tickers"])
    out = {}
    for m, (index, url, col, fix) in PAGES.items():
        tables = pd.read_html(io.StringIO(sd._get(url)))
        t = next(t for t in tables if col in t.columns)
        tickers = list(dict.fromkeys(fix(str(s).strip()) for s in t[col].dropna()))
        if m == "US other":
            tickers = [tk for tk in tickers if tk not in sp100]
        out[m] = {"index": index, "source": url, "tickers": tickers}
        print(f"{m:12s} {index:24s} {len(tickers)}")
    out["_fetched"] = time.strftime("%Y-%m-%d", time.gmtime())
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "constituents.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def fetch_prices():
    import datetime as dt
    cons = json.loads((DATA / "constituents.json").read_text())
    folder = DATA / "prices"
    folder.mkdir(parents=True, exist_ok=True)
    p1 = int(dt.datetime(1999, 12, 1, tzinfo=dt.timezone.utc).timestamp())
    p2 = int(dt.datetime(2026, 10, 29, tzinfo=dt.timezone.utc).timestamp())
    for m in MARKETS:
        for tk in cons[m]["tickers"]:
            f = folder / f"{tk}.csv"
            if f.exists():
                continue
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}?period1={p1}&period2={p2}&interval=1d"
            try:
                res = json.loads(sd._get(url))["chart"]["result"][0]
                off = res["meta"].get("gmtoffset", 0)
                adj = res["indicators"]["adjclose"][0]["adjclose"]
                rows = [(dt.datetime.fromtimestamp(t + off, dt.timezone.utc).strftime("%Y%m%d"), a)
                        for t, a in zip(res.get("timestamp", []), adj) if a is not None and a > 0]
            except Exception as e:  # noqa: BLE001
                print(f"{tk}: {e}")
                rows = []
            f.write_text("date,adjclose\n" + "".join(f"{d},{a:.6g}\n" for d, a in rows))
            time.sleep(0.4)
        print(m, "done")


def _prices(tk):
    a = np.loadtxt(DATA / "prices" / f"{tk}.csv", delimiter=",", skiprows=1, ndmin=2)
    if a.size == 0:
        return np.zeros(0, int), np.zeros(0)
    d, p = a[:, 0].astype(int), a[:, 1]
    _, i = np.unique(d, return_index=True)
    return d[i], p[i]


@lru_cache(maxsize=16)
def panel(market):
    """(dates, returns (days, members) in %, member tickers, info) under e79's panel rule."""
    cons = json.loads((DATA / "constituents.json").read_text())
    series = {tk: _prices(tk) for tk in cons[market]["tickers"] if (DATA / "prices" / f"{tk}.csv").exists()}
    cand = {tk: s for tk, s in series.items()
            if len(s[0]) and s[0][0] <= sd.CAND_FIRST and s[0][-1] >= sd.CAND_LAST}
    if not cand:
        return np.zeros(0, int), np.zeros((0, 0)), [], {"candidates": 0, "members": 0}
    alld = np.unique(np.concatenate([s[0] for s in cand.values()]))
    alld = alld[(alld >= sd.START) & (alld <= sd.END)]
    count = sum(np.isin(alld, s[0]) for s in cand.values())
    cal = alld[count >= 0.5 * len(cand)]
    cover = {tk: np.isin(cal, s[0]).mean() for tk, s in cand.items()}
    members = sorted(tk for tk, c in cover.items() if c >= sd.COVER)
    if len(members) > sd.MAX_MEMBERS:
        members = sorted(np.random.default_rng(sd.SEED).choice(members, sd.MAX_MEMBERS, replace=False).tolist())
    r = np.empty((len(cal) - 1, len(members)))
    for j, tk in enumerate(members):
        d, p = cand[tk]
        idx = np.searchsorted(d, cal, side="right") - 1
        r[:, j] = np.clip(100 * np.diff(np.log(p[np.maximum(idx, 0)])), -sd.CAP, sd.CAP)
    info = {"listed": len(cons[market]["tickers"]), "downloaded": sum(len(s[0]) > 0 for s in series.values()),
            "candidates": len(cand), "members": len(members), "days": len(cal) - 1, "fetched": cons["_fetched"]}
    return cal[1:], r, members, info


if __name__ == "__main__":
    {"constituents": fetch_constituents, "prices": fetch_prices}[sys.argv[1]]()
