"""Individual stocks in eight markets: constituent lists, daily prices and the member panels of e79.

    uv run --with pandas --with lxml python -m experiments.stock_data constituents   # Wikipedia, once
    uv run python -m experiments.stock_data prices                                   # Yahoo Finance, once

Constituents: today's members of eight indices, from Wikipedia on the fetch date (data/stocks/constituents.json):
S&P 100 (US), Nikkei 225 (Japan, Japanese Wikipedia), FTSE 100 (UK), DAX (Germany), CAC 40 (France),
Hang Seng (Hong Kong), S&P/ASX 50 (Australia), S&P/TSX 60 (Canada).
Prices: Yahoo Finance daily split- and dividend-adjusted closes from 2000 (data/stocks/prices/<ticker>.csv).
These are survivors: firms that left the index or were delisted since 2000 are absent.

Panel rule (fixed before any share was computed):
  candidates  first price on or before CAND_FIRST, last price on or after CAND_LAST;
  calendar    days in [START, END] on which at least half of the candidates have a price;
  members     candidates priced on >= COVER of calendar days, at most MAX_MEMBERS (a seeded random subset);
  returns     100 log(P_t / P_prev) on calendar days, P carried forward over a member's missing days, |r| capped
              at CAP (%) against data errors.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path("data/stocks")
START, END = 20000103, 20261023
CAND_FIRST, CAND_LAST = 20000110, 20261016
COVER, MAX_MEMBERS, CAP, SEED = 0.97, 40, 30.0, 79

PAGES = {
    "US": ("S&P 100", "https://en.wikipedia.org/wiki/S%26P_100"),
    "Japan": ("Nikkei 225", "https://ja.wikipedia.org/wiki/%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%E6%A0%AA%E4%BE%A1"),
    "UK": ("FTSE 100", "https://en.wikipedia.org/wiki/FTSE_100_Index"),
    "Germany": ("DAX", "https://en.wikipedia.org/wiki/DAX"),
    "France": ("CAC 40", "https://en.wikipedia.org/wiki/CAC_40"),
    "Hong Kong": ("Hang Seng", "https://en.wikipedia.org/wiki/Hang_Seng_Index"),
    "Australia": ("S&P/ASX 50", "https://en.wikipedia.org/wiki/S%26P%2FASX_50"),
    "Canada": ("S&P/TSX 60", "https://en.wikipedia.org/wiki/S%26P%2FTSX_60"),
}
MARKETS = tuple(PAGES)


def _get(url):
    import subprocess
    out = subprocess.run(["curl", "-sf", "-m", "60", "-A", "Mozilla/5.0 (research; LifeComplexity e79)", url],
                         capture_output=True, check=True)
    return out.stdout.decode("utf-8")


def _tickers(market, html):
    import io

    import pandas as pd
    tables = pd.read_html(io.StringIO(html))
    if market == "Japan":
        codes = [str(c) for t in tables if "証券コード" in t.columns for c in t["証券コード"]]
        return [f"{c}.T" for c in codes]
    col, fix = {
        "US": ("Symbol", lambda s: s.replace(".", "-")),
        "UK": ("Ticker", lambda s: s.rstrip(".").replace(".", "-") + ".L"),
        "Germany": ("Ticker", str),
        "France": ("Ticker", str),
        "Hong Kong": ("Ticker", lambda s: s.split(":")[1].strip().zfill(4) + ".HK"),
        "Australia": ("Symbol", lambda s: s + ".AX"),
        "Canada": ("Symbol", lambda s: s.replace(".", "-") + ".TO"),
    }[market]
    t = max((t for t in tables if col in t.columns), key=len)
    return [fix(str(s).strip()) for s in t[col].dropna()]


def fetch_constituents():
    out = {}
    for m, (index, url) in PAGES.items():
        tickers = list(dict.fromkeys(_tickers(m, _get(url))))
        out[m] = {"index": index, "source": url, "tickers": tickers}
        print(f"{m:10s} {index:12s} {len(tickers)}")
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
                res = json.loads(_get(url))["chart"]["result"][0]
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


@lru_cache(maxsize=8)
def panel(market):
    """(dates, returns (days, members) in %, member tickers, info) under the panel rule."""
    cons = json.loads((DATA / "constituents.json").read_text())
    series = {tk: _prices(tk) for tk in cons[market]["tickers"] if (DATA / "prices" / f"{tk}.csv").exists()}
    cand = {tk: s for tk, s in series.items() if len(s[0]) and s[0][0] <= CAND_FIRST and s[0][-1] >= CAND_LAST}
    alld = np.unique(np.concatenate([s[0] for s in cand.values()])) if cand else np.zeros(0, int)
    alld = alld[(alld >= START) & (alld <= END)]
    count = sum(np.isin(alld, s[0]) for s in cand.values()) if cand else np.zeros(0)
    cal = alld[count >= 0.5 * len(cand)]
    cover = {tk: np.isin(cal, s[0]).mean() for tk, s in cand.items()}
    members = sorted(tk for tk, c in cover.items() if c >= COVER)
    if len(members) > MAX_MEMBERS:
        members = sorted(np.random.default_rng(SEED).choice(members, MAX_MEMBERS, replace=False).tolist())
    r = np.empty((len(cal) - 1, len(members)))
    for j, tk in enumerate(members):
        d, p = cand[tk]
        idx = np.searchsorted(d, cal, side="right") - 1
        lp = np.log(p[np.maximum(idx, 0)])
        r[:, j] = np.clip(100 * np.diff(lp), -CAP, CAP)
    info = {"listed": len(cons[market]["tickers"]), "downloaded": sum(len(s[0]) > 0 for s in series.values()),
            "candidates": len(cand), "qualifying": int(sum(c >= COVER for c in cover.values())), "members": len(members),
            "days": len(cal) - 1, "capped": int((np.abs(r) >= CAP).sum()), "fetched": cons["_fetched"]}
    return cal[1:], r, members, info


if __name__ == "__main__":
    {"constituents": fetch_constituents, "prices": fetch_prices}[sys.argv[1]]()
