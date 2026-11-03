# The procedure: measuring coordination in your field

twinshare computes one number, the **interaction share** S, and checks whether it can be trusted. The number is only
as meaningful as the comparison behind it, and that comparison is a modelling decision only someone who knows the
field can make. This document is that decision, made explicit, in eight steps.

## 1. Is your question a panel question?

You need **members** measured on **one shared clock**: stores × days, sectors × days, neurons × 10 ms bins, sensors ×
minutes. The **group** is the members' average. You are asking:

> Is the group forecast better by its own past than by adding up what each member's own past predicts?

If yes, the members are **coordinated** beyond their own habits and the conditions you recorded. Version 0.1 does not
handle trial-structured data (repeated short episodes), movement trajectories, or a group variable that is not an
average (such as a school's polarization); the lab's experiments show how those were done.

Rough minimums: 10 members, 300 time steps after the longest history window, and series that do not change character
completely halfway through.

## 2. Declare four things before you look at a result

`twinshare init --domain <generic|retail|markets|neuro>` writes a study file with these questions. Answer them
first; the report prints them, and a reader will judge S by them.

| Question | Example (retail) |
| --- | --- |
| What is a member, and the group? | one store's daily sales; the chain's average |
| What would count as interaction? | word of mouth, shared stock-outs, customers switching stores |
| Which shared influences are recorded? | weather, chain-wide promotions, the weekly cycle |
| Which could not be recorded? | local economy, competitor openings, regional events |

## 3. Choose the recorded causes, carefully

The **twin** keeps every member and switches interaction off: each member is forecast from its own past and the
recorded causes only. Anything you record is therefore *never* counted as coordination. Two mistakes:

* **Leaving out a known shared cause.** If every store responds to a promotion you did not record, the group's past
  reveals the promotion better than any single store does, and S counts it as coordination. Record what you know.
* **Recording a consequence of interaction.** A cause must be outside the members. Do not record the group's own
  average, a market index built from the members, a competitor's price that reacts to your stores, or a "regional
  demand" variable estimated from the same sales. Those absorb the coordination you are trying to measure (a "bad
  control"). If in doubt, ask: would this variable change if the members stopped influencing each other? If yes, it
  is not a recorded cause.

Calendar effects (`calendar = "weekday"`, `"month"`) are recorded causes too. twinshare includes every cause both at
the forecast time and over the history windows, because a member's history carries its causes' past effects.

## 4. Choose the member variable, horizon and windows

* **Variable.** Use what is meaningful to average. Sales and counts: `transform = "log"` (with `offset` for zeros).
  Returns: `"logabs"` for volatility (`offset = 0.05` for % returns). Firing rates: as they are.
* **Horizon.** How far ahead the forecast looks, in rows. Coordination that acts within a day is invisible at a
  one-week horizon, and vice versa.
* **Windows.** The memory each series gets: means over the last w steps, ending `horizon` steps back. Cover the
  system's natural scales: `[1, 7, 28]` for daily retail, `[1, 5, 22]` for trading days, `[1, 5, 20]` bins for
  neurons. Both the twin and the group law get the same windows, so this does not favour either side.

Set these before running the full analysis, and do not tune them on S. `twinshare screen` is safe to use while
choosing (it shows only the memory gap and its floor), but treat its number as a preview, not a result.

## 5. Run

```bash
twinshare screen study.toml     # seconds: the memory gap, no fitting
twinshare run study.toml        # the measured share, checks, closure, influence -> report/report.html
```

## 6. Read the checks before the number

| Check | Fails when | What to do |
| --- | --- | --- |
| data | short series, few blocks | treat S as a screen; get more data |
| virtual groups | members shifted against each other still look coordinated | detrend, or record the drift (seasonality, a trend) as a cause |
| null twins | independent members built from your own fitted twins give S above 0 | the measurement is biased at this size; do not interpret |
| power | a known spillover would not be detected | a null result means little; get more data or members |
| memory gap | measured and fit-free predicted S disagree | nonlinear dynamics, rare events or drift: trust the sign more than the size |
| stability | the two halves have opposite signs | coordination is not a stable property of the period: analyse sub-periods |
| block length | S moves with the cross-fitting block length | the series drift; read the shorter-block value, record the drift |

The verdict is "Coordination detected" only if the 90% interval excludes 0, S clears the bar set by the null twins
(their mean plus three spreads), and neither the virtual-group nor the null-twin check failed.

## 7. Interpret within what S can say

* **S is a share of forecast error.** S = 0.05 means the group's own past removes 5% of the error the summed twin
  makes. Markets sit around 0.01–0.03, neurons 0.01–0.07, fish schools 0.1–0.6 (the Emergence Lab's results).
* **S is mechanism-blind.** Interaction and an unrecorded shared driver that each member reveals poorly give the same
  S. Your declarations narrow it down; an intervention decides it (switch interaction off, as in the lab's e80, and
  see S fall).
* **Closure and influence** say *where* the coordination lives. Closure near 0: the group's own past is enough (the
  members respond to the group as a whole). Positive closure, with a few members far ahead in influence: those
  members' pasts lead the group (leaders, early responders). Influence is predictive, not causal.

## 8. Report it so others can check it

Publish the study file, `report.html` and `result.json` together. The report holds the declarations, every check,
the data changes made, the settings, the seed, the twinshare version and the SHA-256 of each input file, so anyone
with the same data and version gets the same numbers. For a confirmatory study, hash the study file and your
predictions before the first full run, as the lab did (`shasum -a 256 study.toml`).

## Domain notes

* **Retail and operations.** Promotions, prices set centrally, holidays and weather are recorded causes; local
  marketing that one store copies from another is interaction. Stores that share a warehouse are coupled through
  stock-outs: decide in advance whether that counts.
* **Markets.** Use volatility (`logabs`) rather than returns: returns are nearly unforecastable, so their share is
  undefined in practice. Market-wide news is the classic unrecorded driver; scheduled announcements can be recorded.
* **Neuroscience.** Bin spikes into rates; record stimulus and movement as causes. Without trials, slow brain-state
  changes act as unrecorded drivers; an optogenetic silencing experiment is the intervention.
* **Ecology and sensors.** Weather, daylight and tides are recorded causes; animals or devices reacting to each other
  are interaction.
