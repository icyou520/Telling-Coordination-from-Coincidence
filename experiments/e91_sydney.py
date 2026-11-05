"""Experiment 91: shared price leadership in Sydney petrol (FuelCheck, September 2016 - December 2017).

Byrne, de Roos, Lewis, Marx and Wu (Informed Sources chapter; FuelCheck data, August 2016 - December 2017) document
18 market-wide restorations in Sydney, about one a month. The five majors (BP, Caltex, Woolworths, Coles Express,
7-Eleven) lead and coordinate them by station-level price signals, and take turns: each signals before 17-39% of
restorations. In their worked example (April 2017) BP jumps first at noon, Woolworths and Caltex within two hours,
7-Eleven by evening, Coles the next day; smaller chains and independents follow below them.
This is the case e90 (Perth) could not cover: e90 found a single leader whose rivals followed a day later. Here the
leaders rotate and most follow within the same day. Development (below) shows that the influence measure cannot name
such leaders at daily resolution: a member's history adds to the group's own past only when that member moves at
least one step before most of the group, and does so consistently. So this experiment tests a stated limit: the
share should see coordination while no brand stands out by influence.

Members: brands (mean U91 price over the brand's Greater Sydney stations, postcodes 2000-2234, 2555-2574 and
2745-2777) minus the AIP's average Sydney terminal gate price, cents per litre. Each station's daily price is the
last price it posted by the end of the day (prices outside 80-250 cpl are dropped as entry errors); a station counts
from its first posted price, and August 2016 is a warm-up for that. A brand is a member if it has at least 10 stations
on the median day and at least one on 99% of days. Recorded causes: the day's change in terminal gate price (carried
over weekends and holidays) and weekday dummies. Design, measures and virtual groups as e90 (horizon 1 day, windows
1-7, 14, 28 days; whole-week records and shifts). Majors (fixed from the literature): BP, Caltex, Caltex Woolworths,
Coles Express, 7-Eleven.

Development (no real share was computed). First with five majors and five independents (seed 0, 518 days):
  * independent firms: S -0.011, every influence <= 0.005;
  * one firm leads every restoration and all others follow a day or more later: leader +0.68;
  * the same leader, but most majors follow within the day: leader +0.041;
  * majors take turns, others follow within the day, independents 1-3 days later: S +0.43, every influence <= 0.013,
    and the majors as a block add +0.003 (the group's own average already shows the restoration).
Then the simulator below, with Sydney's twelve brands and 490 days, on development seeds 0 and 10-13:
  * independent: |S| <= 0.030, every influence <= 0.023 (virtual groups included);
  * leader: leader largest, +0.192..+0.457;
  * leader_sameday and rotating: S +0.37..+0.62, largest influence +0.021..+0.111 (in rotating, usually Caltex, the
    most frequent co-leader); leader_sameday's leader once -2.2 (a rare-event overfit, not a pass or fail matter);
  * virtual groups: S <= 0.052, largest influence <= 0.135. On 70 weeks the noise floor is higher than in Perth
    (e90, 190 weeks: <= 0.045), so the line for naming a leader here is 0.15. Perth's BP (+0.215) would clear it.
  * the structural smoke test (seed 101) gave independent firms S -0.040, so P1 allows |S| <= 0.06 as P4 does.

Pre-registered predictions (simulations seeds 1-3; real period run once):
  P1  Independent firms: |S| <= 0.06 and every influence < 0.03 (each seed).
  P2  One leader, followed a day or more later: the leader has the largest influence and it exceeds 0.15 (each seed).
  P3  Leadership the measure cannot see (rotating majors with same-day following; one leader with same-day following
      by most majors): S > 0.1 and every influence < 0.15 (each seed).
  P4  Virtual groups, every simulation and the real period: S <= 0.06 and largest influence < 0.15.
  P5  Sydney 2016-2017: S > 0.05 and the lower end of its interval > 0 (coordination is seen).
  P6  Sydney 2016-2017: every brand's influence < 0.15 (no leader is named).
Reported, not predicted: the majors' block gain against the other members' block, margins and their variation.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twinshare" / "src"))
from twinshare import engine as E  # noqa: E402

from experiments.e90_petrol import DESIGN, measure  # noqa: E402
from experiments import fuelcheck_data as F  # noqa: E402

START, END, WARMUP = "2016-09-01", "2017-12-31", "2016-08-01"
MAJORS = ("BP", "Caltex", "Caltex Woolworths", "Coles Express", "7-Eleven")
MIN_STATIONS, MIN_PRESENT = 10, 0.99
PRICE_RANGE = (80.0, 250.0)
LINE = 0.15                                    # influence needed to name a leader on a 70-week record


def sydney_postcode(pc):
    return pc.isdigit() and (2000 <= int(pc) <= 2234 or 2555 <= int(pc) <= 2574 or 2745 <= int(pc) <= 2777)


# ---------------------------------------------------------------- real data

def sydney(start=START, end=END):
    """Brand margins (T, N), recorded causes (T, 7), dates, member names, stations per member (median day)."""
    events = defaultdict(list)                                   # (brand, address) -> [(time, price)]
    for ym in F.MONTHS:
        for r in F.retail(ym):
            p = float(r["price"])
            if sydney_postcode(r["postcode"]) and PRICE_RANGE[0] <= p <= PRICE_RANGE[1] and r["time"] <= end + " 23:59:59":
                events[(r["brand"], r["address"].lower())].append((r["time"], p))
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    dates = [(d0 + dt.timedelta(k)).isoformat() for k in range((d1 - d0).days + 1)]
    price = defaultdict(lambda: defaultdict(list))                # brand -> date -> station prices
    for (brand, _), ev in events.items():
        ev.sort()
        k, last = 0, None
        for d in dates:
            while k < len(ev) and ev[k][0] <= d + " 23:59:59":
                last, k = ev[k][1], k + 1
            if last is not None:
                price[brand][d].append(last)
    members = sorted(b for b, by in price.items()
                     if np.median([len(by.get(d, ())) for d in dates]) >= MIN_STATIONS
                     and sum(d in by for d in dates) >= MIN_PRESENT * len(dates))
    tgp, cost = F.tgp_sydney(), []
    for d in dates:
        back = dt.date.fromisoformat(d)
        while back.isoformat() not in tgp:
            back -= dt.timedelta(1)
        cost.append(tgp[back.isoformat()])
    cost = np.array(cost)
    x = np.column_stack([[np.mean(price[b][d]) for d in dates] for b in members]) - cost[:, None]
    dow = np.array([dt.date.fromisoformat(d).weekday() for d in dates])
    z = np.column_stack([np.diff(cost, prepend=cost[0]), np.eye(7)[dow][:, 1:]])
    stations = [float(np.median([len(price[b][d]) for d in dates])) for b in members]
    return {"x": x, "z": z, "dates": dates, "members": members, "stations": stations}


# ---------------------------------------------------------------- simulations

FIRMS_MAJOR = (81, 127, 56, 62, 138)          # BP, Caltex, Caltex Woolworths, Coles Express, 7-Eleven
FIRMS_OTHER = (86, 72, 39, 34, 23, 13, 12)     # Metro, Independent, Budget, Speedway, United, Mobil, Westside
DAYS = 490
TOP, FLOOR = 22.0, 0.0
REGIMES = ("independent", "leader", "leader_sameday", "rotating")


def simulate(regime, rng, days=DAYS, majors=FIRMS_MAJOR, others=FIRMS_OTHER):
    """Station-level margins averaged by firm, with the wholesale cost change and weekday as recorded causes.

    independent     each firm restores on its own timer (20-40 days), ignoring rivals;
    leader          firm 0 restores when the majors' margins are low; every other firm follows 1-3 days later;
    leader_sameday  as leader, but each other major follows the same day with probability 0.8 (else next day);
    rotating        a major chosen with weights 0.3/0.3/0.25/0.15/0 (BP, Caltex, Woolworths, Coles, 7-Eleven)
                    restores; other majors except Coles follow the same day with probability 0.8 (else next day),
                    Coles the next day, other firms 1-3 days later.
    Between restorations stations cut 0.3-1 cpl a day and undercut the median firm.
    """
    firms = tuple(majors) + tuple(others)
    f, nm = len(firms), len(majors)
    owner = np.repeat(np.arange(f), firms)
    n = len(owner)
    offset = rng.normal(0, 1.5, n) + np.where(owner < nm, 1.0, -1.5)
    m = TOP / 2 + offset
    dc = np.zeros(days)
    out = np.zeros((days, f))
    pending = {}
    timer = rng.integers(0, 30, f)
    weights = np.array([0.3, 0.3, 0.25, 0.15, 0.0])
    for t in range(days):
        dc[t] = 0.3 * dc[t - 1] + rng.normal(0, 0.8) if t else 0.0
        firm_m = np.array([m[owner == k].mean() for k in range(f)])
        new = m - rng.uniform(0.3, 1.0) - 0.6 * dc[t] + rng.normal(0, 0.5, n)
        jump_firm = np.zeros(f, bool)
        if regime == "independent":
            timer -= 1
            jump_firm = (timer <= 0) | (firm_m < FLOOR - 2)
            timer = np.where(jump_firm, rng.integers(20, 40, f), timer)
        else:
            new = np.minimum(new, np.median(firm_m) - 0.3 + offset + rng.normal(0, 0.4, n))
            if t in pending:
                jump_firm[pending.pop(t)] = True
            elif firm_m[:nm].mean() < FLOOR + 3 and rng.random() < 0.25 and not pending:
                lead = rng.choice(nm, p=weights / weights.sum()) if regime == "rotating" else 0
                same = [] if regime == "leader" else \
                    [k for k in range(nm) if k != lead and not (regime == "rotating" and k == 3)
                     and rng.random() < 0.8]
                jump_firm[[lead] + same] = True
                for k in range(nm):
                    if k != lead and k not in same:
                        pending.setdefault(t + (1 if regime != "leader" else int(rng.integers(1, 4))), []).append(k)
                for k in range(nm, f):
                    pending.setdefault(t + int(rng.integers(1, 4)), []).append(k)
        m = np.where(jump_firm[owner], TOP + offset + rng.normal(0, 0.5, n), np.maximum(new, FLOOR - 6 + offset))
        out[t] = [m[owner == k].mean() for k in range(f)]
    z = np.column_stack([dc, np.eye(7)[np.arange(days) % 7][:, 1:]])
    return out, z


# ---------------------------------------------------------------- predictions and report

SEEDS = (1, 2, 3)


def _sims(res, name):
    return [r for k, r in res.items() if k.startswith(f"sim {name}, seed")]


def predictions(res):
    out = []
    ind = _sims(res, "independent")
    out.append(("P1", "independent firms: |S| <= 0.06, every influence < 0.03",
                all(abs(r["S"]) <= 0.06 and max(r["influence"]) < 0.03 for r in ind),
                ", ".join(f"S {r['S']:+.3f} max {max(r['influence']):+.3f}" for r in ind)))
    lead = [r for k, r in res.items() if k.startswith("sim leader, seed")]
    out.append(("P2", "one leader followed a day or more later: leader largest, > 0.15",
                all(int(np.argmax(r["influence"])) == 0 and r["influence"][0] > LINE for r in lead),
                ", ".join(f"{r['influence'][0]:+.3f}" for r in lead)))
    blind = _sims(res, "rotating") + _sims(res, "leader_sameday")
    out.append(("P3", "rotating or same-day leadership: S > 0.1, every influence < 0.15",
                all(r["S"] > 0.1 and max(r["influence"]) < LINE for r in blind),
                ", ".join(f"S {r['S']:+.3f} max {max(r['influence']):+.3f}" for r in blind)))
    vs = [v for r in res.values() for v in r["virtual"]]
    out.append(("P4", "virtual groups: S <= 0.06 and largest influence < 0.15",
                all(v["S"] <= 0.06 and v["influence_max"] < LINE for v in vs),
                f"{len(vs)} groups, max S {max(v['S'] for v in vs):+.3f}, "
                f"max influence {max(v['influence_max'] for v in vs):+.3f}"))
    syd = res.get("Sydney 2016-2017")
    if syd:
        out.append(("P5", "Sydney: S > 0.05, interval lower end > 0",
                    syd["S"] > 0.05 and syd["interval"][0] > 0,
                    f"S {syd['S']:+.3f} [{syd['interval'][0]:+.3f}, {syd['interval'][1]:+.3f}]"))
        top = int(np.argmax(syd["influence"]))
        out.append(("P6", "Sydney: every brand's influence < 0.15", max(syd["influence"]) < LINE,
                    f"max {syd['members'][top]} {max(syd['influence']):+.3f}"))
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
        lines.append(f"    stations (median day): " + ", ".join(f"{m} {s:.0f}" for m, s in zip(r["members"],
                                                                                          r["stations"])))
        lines.append(f"    block gain: majors {r['block_majors']:+.3f}, others {r['block_others']:+.3f}")
        lines.append(f"    mean margin {r['margin_mean']:.2f} cpl, coefficient of variation {r['margin_cv']:.3f}")
    return lines


def analyze(res):
    lines = ["Experiment 91: shared price leadership in Sydney petrol (FuelCheck).", ""]
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
        for i, regime in enumerate(REGIMES):
            rng = np.random.default_rng([91, s, i])
            x, z = simulate(regime, rng)
            res[f"sim {regime}, seed {s}"] = measure(x, z, rng)
    if not quick:
        p = sydney()
        r = measure(p["x"], p["z"], np.random.default_rng([91, 1000]))
        k = len(p["x"]) % 7
        x, z = p["x"][k:], p["z"][k:]
        maj = [i for i, m in enumerate(p["members"]) if m in MAJORS]
        oth = [i for i in range(len(p["members"])) if i not in maj]
        r.update(members=p["members"], stations=p["stations"], first=p["dates"][k], last=p["dates"][-1],
                 block_majors=E.gain(x, z, DESIGN, maj), block_others=E.gain(x, z, DESIGN, oth))
        res["Sydney 2016-2017"] = r
    (out_dir / "e91_sydney_raw.json").write_text(json.dumps(res))
    text = analyze(res)
    from experiments.e91_figures import make_figures
    make_figures(res, out_dir / "e91_figures", text)
    return text


if __name__ == "__main__":
    main()
