"""Study files: the question, the domain assumptions that make the twin trustworthy, the data and the settings.

A study is a TOML file with four tables. [study] is for people: what a member is, what would count as interaction,
which shared influences are recorded (the twin may use them, so they are never counted as coordination) and which
are not (a positive share may come from them). These declarations are printed at the top of every report, because
the share is only as meaningful as they are. [data], [model] and [run] are for the tool.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .engine import Design

DECLARATIONS = ("title", "members", "group", "interaction", "recorded_causes", "unrecorded")

TEMPLATE = '''# twinshare study file. Fill in [study] before running: the report prints it, and a reader judges the result
# by it. See PROCEDURE.md for how to answer each question in your field.

[study]
title = "{title}"
analyst = ""
# What is one member, and what is the group (the members' average)?
members = "{members}"
group = "{group}"
# What would count as interaction between members in this field?
interaction = "{interaction}"
# Shared influences that are NOT interaction and that you have as data (columns in [data].covariates, or the
# calendar). The twin uses them, so they never count as coordination.
recorded_causes = "{recorded}"
# Shared influences you could not record. A positive share can come from these instead of interaction; only an
# intervention or recording them can tell.
unrecorded = "{unrecorded}"

[data]
file = "{file}"
format = "{format}"            # "wide": time column + one column per member; "long": time, member, value columns
time = "{time}"
{long_cols}covariates = {covariates}   # recorded causes in the same file (wide) or in covariates_file
# covariates_file = "causes.csv"   # optional: recorded causes keyed by the same time column
max_missing = 0.1               # drop members missing more than this share of time steps

[model]
transform = "{transform}"          # none, log (x + offset), logabs (|x| + offset)
offset = {offset}
calendar = "{calendar}"         # none, weekday, month, weekday+month (needs dates in the time column)
horizon = {horizon}                     # forecast this many steps ahead
windows = {windows}           # history windows (steps) for every series: the twin's and the group law's memory

[run]
seed = 0
'''

DOMAINS = {
    "generic": dict(
        title="Does the group move together beyond what its members and the recorded causes explain?",
        members="One unit's measurement over time", group="The average across units",
        interaction="Units influencing each other directly", recorded="(list the shared conditions you have as data)",
        unrecorded="(list shared conditions you could not record)", file="data.csv", format="wide", time="time",
        covariates="[]", transform="none", offset=0.0, calendar="none", horizon=1, windows="[1, 5, 20]"),
    "retail": dict(
        title="Do our stores' daily sales move together beyond weather, promotions and the weekly cycle?",
        members="One store's daily sales", group="Average daily sales across the chain",
        interaction="Stores affecting each other: word of mouth, shared stock-outs, customers switching stores",
        recorded="Weather, chain-wide promotions, the weekly cycle",
        unrecorded="Local economy, competitor openings, regional events, payday effects",
        file="sales.csv", format="wide", time="date", covariates='["temperature", "promotion"]', transform="log",
        offset=1.0, calendar="weekday", horizon=1, windows="[1, 7, 28]"),
    "markets": dict(
        title="Is market volatility shaped by coordination between its sectors?",
        members="One sector's daily absolute return (volatility proxy)", group="The market's average",
        interaction="Spillovers: sectors reacting to each other's shocks, contagion, cross-holdings",
        recorded="The weekly cycle (add macro announcements or market-wide news if you have them)",
        unrecorded="Market-wide news and macro shocks that are not in the data",
        file="returns.csv", format="wide", time="date", covariates="[]", transform="logabs", offset=0.05,
        calendar="weekday", horizon=1, windows="[1, 5, 22]"),
    "neuro": dict(
        title="Do these neurons coordinate beyond their own dynamics and the task?",
        members="One neuron's binned firing rate", group="The population's mean rate",
        interaction="Synaptic or circuit coupling between the recorded neurons",
        recorded="Task events and behaviour (stimulus, movement), as covariate columns",
        unrecorded="Inputs from unrecorded areas, arousal, brain state",
        file="rates.csv", format="wide", time="t", covariates="[]", transform="none", offset=0.0, calendar="none",
        horizon=1, windows="[1, 5, 20]"),
}


def template(domain: str = "generic") -> str:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {', '.join(DOMAINS)}")
    v = dict(DOMAINS[domain])
    long_cols = "" if v["format"] == "wide" else 'member = "member"\nvalue = "value"\n'
    return TEMPLATE.format(long_cols=long_cols, **v)


def load_study(path: Path) -> dict:
    with open(path, "rb") as f:
        s = tomllib.load(f)
    for key in ("study", "data"):
        if key not in s:
            raise ValueError(f"{path}: missing [{key}] table")
    blank = [k for k in DECLARATIONS if not str(s["study"].get(k, "")).strip()
             or str(s["study"].get(k, "")).startswith("(")]
    s["_undeclared"] = blank
    return s


def design(study: dict) -> Design:
    m = study.get("model", {})
    return Design(horizon=int(m.get("horizon", 1)), windows=tuple(int(w) for w in m.get("windows", (1, 5, 22))),
                  block=m.get("block"), folds=int(m.get("folds", 3)))
