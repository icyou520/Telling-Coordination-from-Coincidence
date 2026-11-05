"""Experiment 90: price leadership and tacit collusion in Perth petrol (FuelWatch, 2004-2026).

Byrne and de Roos (2019, AER 109, 591-619) document, from the same public data, how BP initiated tacit collusion:
    2004-2008  a price cycle (jumps on varying days, 7-14-day cycles); BP and Caltex lead jumps, after 2006 BP alone
               with Caltex following by a day;
    2009-2012  BP restarts the cycle (April 2009) with Wednesday jumps, first with all its stations, then a shrinking
               subset; the rest of the market jumps on Thursday;
    2012-2015  from mid-2012 no BP station jumps on Wednesday: every firm jumps on Thursday and cuts 2 cpl a day.
The question: does the share, with member influence, see the leader, and what does it see once collusion is a
calendar rule? In simulations (development, below) a leader whose rivals follow it by a day stands out by influence,
while firms that all jump on the same weekday, reacting to nobody, read as no coordination: the weekday is a
recorded cause. So the prediction is that leadership is visible in 2004-2011 and fades by 2012-2015.

Members: brands (mean ULP price over the brand's Perth metro stations) minus the wholesale terminal gate price
(median over the Kwinana and Kewdale terminals), cents per litre. A brand is a member in a period if it has at least
10 metro stations on the median day and at least one station on 99% of days; remaining gaps are carried forward.
Recorded causes: the day's change in terminal gate price and weekday dummies. Design: horizon 1 day, windows
(1, 2, 3, 4, 5, 6, 7, 14, 28) days. Influence: twinshare's per-member closure gain.

Periods (fixed from the published narrative before any share was computed):
    A 2004-09-01..2008-05-31  cycle led by BP (and Caltex before 2006);
    B 2009-05-01..2011-12-31  BP Wednesday leadership;
    C 2012-07-01..2015-05-31  mature Thursday focal point;
    D 2016-01-01..2019-12-31, E 2020-01-01..2022-12-31, F 2024-01-01..2026-10-31  never analysed in the literature
      (exploratory; 2023 is skipped because Caltex became Ampol and Coles Express became Reddy Express).
Every record is trimmed to whole weeks, and virtual groups shift each member by whole weeks (see shift_weeks).

Development (simulations seed 0, 1,000 days, 8 firms with period B's station counts; no real share was computed):
  * independent firms: S -0.018, every influence <= 0.001;
  * cycle led by firm 0: S +0.056 (S_pred +0.068), firm 0's influence +0.463, others <= 0.018;
  * cycle where firm 1 leads a third of restorations: firm 0 +0.151, firm 1 +0.012;
  * Thursday focal point: S -0.001, every influence <= 0.000;
  * Wednesday leadership by a share of firm 0's stations: 100% +0.620, 60% +0.304, 20% -0.034. Leadership on a
    fixed weekday by a few stations is mostly explained by the weekday itself;
  * virtual groups: day-level shifts read up to 0.39 on weekly cycles (fixed by whole-week shifts and records);
    with them, virtual S <= +0.011 and influence <= 0.023.
  Because BP's Wednesday leaders fell from its whole network (April-August 2009) to 12-14 stations (to mid-2012),
  development cannot say what period B should show: it is reported, not predicted.

Pre-registered predictions (simulations seeds 1-3; real periods A-F, run once):
  P1  Independent firms: |S| <= 0.03 and every influence < 0.02 (each seed).
  P2  Cycle led by firm 0: firm 0 has the largest influence and it exceeds 0.2 (each seed); with shared leadership,
      firm 0 still has the largest influence and it exceeds 0.05.
  P3  Calendar rules: Thursday focal point, every influence < 0.02 and S <= 0.02; Wednesday leadership by 20% of the
      leader's stations, leader influence < 0.05; by all of them, leader the largest and > 0.2 (each seed).
  P4  Virtual groups, every simulation and every real period: S <= 0.02 and largest influence < 0.04.
  P5  Perth 2004-2008 (A): BP has the largest influence of all brands, and it exceeds 0.05.
  P6  Perth 2012-2015 (C): every brand's influence < 0.05 (a calendar rule shows no leader).
  P7  BP's influence in A exceeds its influence in C.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twinshare" / "src"))
from twinshare import engine as E  # noqa: E402

ROOT = Path("data/fuelwatch")
DESIGN = E.Design(windows=(1, 2, 3, 4, 5, 6, 7, 14, 28))
MIN_STATIONS, MIN_PRESENT = 10, 0.99
TERMINALS = ("Kwinana", "Kewdale")
PERIODS = {
    "A 2004-2008 cycle": ("2004-09-01", "2008-05-31", "known"),
    "B 2009-2011 Wednesday leader": ("2009-05-01", "2011-12-31", "known"),
    "C 2012-2015 Thursday focal": ("2012-07-01", "2015-05-31", "known"),
    "D 2016-2019": ("2016-01-01", "2019-12-31", "exploratory"),
    "E 2020-2022": ("2020-01-01", "2022-12-31", "exploratory"),
    "F 2024-2026": ("2024-01-01", "2026-10-31", "exploratory"),
}


# ---------------------------------------------------------------- real data

def _months(start, end):
    y, m = int(start[:4]), int(start[5:7])
    while (y, m) <= (int(end[:4]), int(end[5:7])):
        yield f"{y:04d}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _tgp(start, end):
    out = defaultdict(list)
    for y in range(int(start[:4]), int(end[:4]) + 1):
        with gzip.open(ROOT / "wholesale" / f"{y}.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                if start <= r["date"] <= end and any(k in r["terminal"] for k in TERMINALS):
                    out[r["date"]].append(float(r["price"]))
    return {d: float(np.median(v)) for d, v in out.items()}


def perth(start, end):
    """Brand margins (T, N), recorded causes (T, 7), dates, member names, stations per day per member."""
    price = defaultdict(lambda: defaultdict(list))            # brand -> date -> station prices
    for ym in _months(start, end):
        with gzip.open(ROOT / "retail" / f"{ym}.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                if r["region"] == "Metro" and start <= r["date"] <= end:
                    price[r["brand"]][r["date"]].append(float(r["price"]))
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    dates = [(d0 + dt.timedelta(k)).isoformat() for k in range((d1 - d0).days + 1)]
    members = sorted(b for b, by in price.items()
                     if np.median([len(by.get(d, ())) for d in dates]) >= MIN_STATIONS
                     and sum(d in by for d in dates) >= MIN_PRESENT * len(dates))
    tgp = _tgp(start, end)
    cols, filled = [], 0
    for series in [[np.mean(price[b][d]) if d in price[b] else np.nan for d in dates] for b in members] + \
                  [[tgp.get(d, np.nan) for d in dates]]:
        s = np.array(series)
        for k in range(1, len(s)):
            if np.isnan(s[k]):
                s[k], filled = s[k - 1], filled + 1
        cols.append(s)
    a = np.column_stack(cols)
    keep = ~np.isnan(a).any(1)
    a, dates = a[keep], [d for d, k in zip(dates, keep) if k]
    x, cost = a[:, :-1] - a[:, -1:], a[:, -1]
    dow = np.array([dt.date.fromisoformat(d).weekday() for d in dates])
    z = np.column_stack([np.diff(cost, prepend=cost[0]), np.eye(7)[dow][:, 1:]])
    stations = np.array([np.median([len(price[b].get(d, ())) for d in dates]) for b in members])
    return {"x": x, "z": z, "dates": dates, "members": members, "stations": stations, "filled": filled}


# ---------------------------------------------------------------- simulations

FIRMS = (66, 43, 43, 49, 36, 14, 13, 18)    # stations per brand in period B; firm 0 is the leader (BP), 1 Caltex
DAYS = 1000
TOP, FLOOR = 18.0, 3.0                       # margin after a jump, and where cycles bottom out (cpl)
REGIMES = ("independent", "cycle", "cycle_shared", "wednesday", "focal")


def simulate(regime, rng, days=DAYS, firms=FIRMS, subset=0.3):
    """Station-level margins averaged by firm, with the wholesale cost change and weekday as recorded causes.

    independent   each firm cuts and restores on its own timer, ignoring rivals (no interaction);
    cycle         firms undercut the cheapest rival; the leader restores when margins are low and the others
                  follow one day later (two days for a few);
    cycle_shared  as cycle, but firm 1 also leads on a third of restorations;
    wednesday     a subset of the leader's stations jumps on Wednesday; the market (and the rest of the leader)
                  jumps on Thursday if the leader jumped, otherwise rarely; 2 cpl daily cuts;
    focal         every firm jumps on Thursday and cuts 2 cpl a day, reacting to nobody.
    """
    owner = np.repeat(np.arange(len(firms)), firms)
    n, f = len(owner), len(firms)
    offset = rng.normal(0, 1.0, n)
    m = TOP / 2 + offset
    dc = np.zeros(days)
    out = np.zeros((days, f))
    lead_today = np.zeros(f, bool)
    timer = rng.integers(0, 10, f)
    for t in range(days):
        dow = t % 7
        dc[t] = 0.3 * dc[t - 1] + rng.normal(0, 0.8) if t else 0.0
        firm_m = np.array([m[owner == k].mean() for k in range(f)])
        cut = 2.0 if regime in ("wednesday", "focal") else rng.uniform(1.0, 2.5)
        new = m - cut - 0.6 * dc[t] + rng.normal(0, 0.6, n)
        jump_firm = np.zeros(f, bool)
        jump_station = np.zeros(n, bool)
        if regime == "independent":
            timer -= 1
            jump_firm = (timer <= 0) | (firm_m < FLOOR)
            timer = np.where(jump_firm, rng.integers(6, 13, f), timer)
        elif regime in ("cycle", "cycle_shared"):
            new = np.minimum(new, firm_m.min() - 0.4 + offset + rng.normal(0, 0.4, n))
            lead = np.zeros(f, bool)
            if firm_m.mean() < FLOOR + 3 and rng.random() < 0.5:
                lead[1 if regime == "cycle_shared" and rng.random() < 1 / 3 else 0] = True
            follow = lead_today.any() & ~lead_today & (rng.random(f) < 0.85)
            late = (firm_m < FLOOR + 1) & (rng.random(f) < 0.3)
            jump_firm = lead | follow | late
            lead_today = lead
        elif regime == "wednesday":
            if dow == 2 and rng.random() < 0.9:
                jump_station = (owner == 0) & (rng.random(n) < subset)
                lead_today = np.array([True] + [False] * (f - 1))
            elif dow == 3:
                jump_firm = rng.random(f) < (0.95 if lead_today[0] else 0.15)
                jump_station = (owner == 0) & ~(m > TOP - 4)
                jump_station &= jump_firm[0]
                jump_firm[0] = False
                lead_today[:] = False
        elif regime == "focal":
            jump_firm = (dow == 3) & (rng.random(f) < 0.95)
        jump = jump_firm[owner] | jump_station
        m = np.where(jump, TOP + offset + rng.normal(0, 0.5, n), np.maximum(new, FLOOR - 2 + offset))
        out[t] = [m[owner == k].mean() for k in range(f)]
    z = np.column_stack([dc, np.eye(7)[np.arange(days) % 7][:, 1:]])
    return out, z


# ---------------------------------------------------------------- measures

def shift_weeks(x, rng):
    """Virtual group: each member circularly shifted by its own whole number of weeks (x must span whole weeks).
    Shifts of other lengths, or a record that is not whole weeks long, move part of a weekly cycle off the weekday
    it follows, which reads as coordination (development: virtual shares up to 0.39)."""
    t, n = x.shape
    assert t % 7 == 0
    weeks = t // 7
    m = max(weeks // 20, 1)
    off = 7 * rng.integers(m, weeks - m + 1, n)
    idx = (np.arange(t)[:, None] + off[None]) % t
    return x[idx, np.arange(n)[None]]


def measure(x, z, rng, n_virtual=5):
    """Share, memory gap, influence and virtual groups, on the last whole number of weeks."""
    k = len(x) % 7
    x, z = x[k:], z[k:]
    err = E.share(x, z, DESIGN)
    inf = E.influence(x, z, DESIGN)
    virt = []
    for _ in range(n_virtual):
        xs = shift_weeks(x, rng)
        virt.append({"S": E.share(xs, z, DESIGN).S, "influence_max": float(E.influence(xs, z, DESIGN).max())})
    return {"S": err.S, "interval": list(err.interval(rng)), "S_pred": E.memory_gap(x, z, DESIGN)["S_pred"],
            "influence": inf.tolist(), "virtual": virt, "days": len(x),
            "margin_mean": float(x.mean()), "margin_cv": float(x.mean(1).std() / abs(x.mean()))}


# ---------------------------------------------------------------- predictions and report

SEEDS = (1, 2, 3)
SIMS = {"independent": ("independent", {}), "cycle": ("cycle", {}), "cycle shared": ("cycle_shared", {}),
        "focal": ("focal", {}), "wednesday 20%": ("wednesday", {"subset": 0.2}),
        "wednesday 100%": ("wednesday", {"subset": 1.0})}


def _sims(res, name):
    return [r for k, r in res.items() if k.startswith(f"sim {name}, seed")]


def _top(r):
    return int(np.argmax(r["influence"]))


def predictions(res):
    out = []
    ind = _sims(res, "independent")
    out.append(("P1", "independent firms: |S| <= 0.03, every influence < 0.02",
                all(abs(r["S"]) <= 0.03 and max(r["influence"]) < 0.02 for r in ind),
                ", ".join(f"S {r['S']:+.3f} max {max(r['influence']):+.3f}" for r in ind)))
    cyc, sh = _sims(res, "cycle"), _sims(res, "cycle shared")
    out.append(("P2", "cycle: leader largest > 0.2; shared leadership: leader largest > 0.05",
                all(_top(r) == 0 and r["influence"][0] > 0.2 for r in cyc)
                and all(_top(r) == 0 and r["influence"][0] > 0.05 for r in sh),
                "; ".join(f"{r['influence'][0]:+.3f}" for r in cyc + sh)))
    foc, w20, w100 = _sims(res, "focal"), _sims(res, "wednesday 20%"), _sims(res, "wednesday 100%")
    out.append(("P3", "calendar rules: focal no leader (< 0.02, S <= 0.02); Wednesday 20% < 0.05; 100% largest > 0.2",
                all(max(r["influence"]) < 0.02 and r["S"] <= 0.02 for r in foc)
                and all(r["influence"][0] < 0.05 for r in w20)
                and all(_top(r) == 0 and r["influence"][0] > 0.2 for r in w100),
                f"focal max {max(max(r['influence']) for r in foc):+.3f}; 20% "
                + ", ".join(f"{r['influence'][0]:+.3f}" for r in w20) + "; 100% "
                + ", ".join(f"{r['influence'][0]:+.3f}" for r in w100)))
    vs = [v for r in res.values() for v in r["virtual"]]
    out.append(("P4", "virtual groups: S <= 0.02 and largest influence < 0.04",
                all(v["S"] <= 0.02 and v["influence_max"] < 0.04 for v in vs),
                f"{len(vs)} groups, max S {max(v['S'] for v in vs):+.3f}, "
                f"max influence {max(v['influence_max'] for v in vs):+.3f}"))
    a, c = res.get(f"Perth {list(PERIODS)[0]}"), res.get(f"Perth {list(PERIODS)[2]}")
    if a and c:
        bp_a, bp_c = a["influence"][a["members"].index("BP")], c["influence"][c["members"].index("BP")]
        out.append(("P5", "Perth 2004-2008: BP largest influence, > 0.05",
                    a["members"][_top(a)] == "BP" and bp_a > 0.05,
                    f"BP {bp_a:+.3f}; top {a['members'][_top(a)]} {max(a['influence']):+.3f}"))
        out.append(("P6", "Perth 2012-2015: every influence < 0.05", max(c["influence"]) < 0.05,
                    f"max {c['members'][_top(c)]} {max(c['influence']):+.3f}"))
        out.append(("P7", "BP influence 2004-2008 > 2012-2015", bp_a > bp_c, f"{bp_a:+.3f} vs {bp_c:+.3f}"))
    return out


def describe(name, r):
    inf = np.array(r["influence"])
    names = r.get("members") or [f"firm {i}" for i in range(len(inf))]
    order = np.argsort(-inf)
    vs = [v["S"] for v in r["virtual"]]
    vi = [v["influence_max"] for v in r["virtual"]]
    lines = [f"  {name}  ({r['days']} days)",
             f"    S {r['S']:+.4f} [{r['interval'][0]:+.4f}, {r['interval'][1]:+.4f}]  S_pred {r['S_pred']:+.4f}"
             f"  virtual S {min(vs):+.4f}..{max(vs):+.4f}, influence <= {max(vi):+.4f}",
             "    influence: " + ", ".join(f"{names[i]} {inf[i]:+.3f}" for i in order)]
    if "members" in r:
        lines.append(f"    mean margin {r['margin_mean']:.2f} cpl, coefficient of variation {r['margin_cv']:.3f}")
    return lines


def analyze(res):
    lines = ["Experiment 90: price leadership and tacit collusion in Perth petrol (FuelWatch).", ""]
    for k, r in res.items():
        lines += describe(k, r)
    lines += ["", "  Pre-registered predictions:"]
    for key, label, ok, detail in predictions(res):
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {key} {label} [{detail}]")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    """quick: simulations only, one fresh seed, no real data (a structural smoke test)."""
    out_dir = out_dir or Path("results")
    res = {}
    for s in ((101,) if quick else SEEDS):
        for name, (regime, kw) in SIMS.items():
            rng = np.random.default_rng([90, s, list(SIMS).index(name)])
            x, z = simulate(regime, rng, **kw)
            res[f"sim {name}, seed {s}"] = measure(x, z, rng)
    if not quick:
        for name, (start, end, _) in PERIODS.items():
            p = perth(start, end)
            r = measure(p["x"], p["z"], np.random.default_rng([90, 1000, list(PERIODS).index(name)]))
            r.update(members=p["members"], stations=p["stations"].tolist(), first=p["dates"][0], last=p["dates"][-1])
            res[f"Perth {name}"] = r
    (out_dir / "e90_petrol_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e90_figures import make_figures
    make_figures(res, out_dir / "e90_figures", text)
    return text


if __name__ == "__main__":
    main()
