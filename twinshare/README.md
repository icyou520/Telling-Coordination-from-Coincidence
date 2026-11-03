# twinshare

**Is a group coordinated beyond what its members and the recorded conditions explain?**

Fifty stores' daily sales move together. Is that because the stores influence each other, or only because they share
the weather, the promotions and the weekly cycle? twinshare answers with the **interaction share**: it builds a
*no-interaction twin* (every member kept, each forecast from its own past and the recorded causes only) and measures
how much better the group's own past forecasts the group. Then it checks whether that number can be trusted.

```
$ twinshare example retail --coupling 0.3 --out demo && twinshare run demo/study.toml
Coordination detected (strong)
The group is forecast better by its own history than by adding up its members' own forecasts: S = +0.351
(90% interval +0.280 to +0.409). ...

  pass  data: 1067 rows in 20 blocks, 50 members
  pass  virtual groups: mean S -0.0056 (range -0.0120 to -0.0017) over 5 shifted groups
  pass  null twins: mean -0.0084, spread 0.0078 over 8 independent simulations; coordination needs S > +0.0150
  pass  power: known spillover gives S +0.0696; detected in 2/2
  pass  memory gap: measured +0.3508, predicted +0.2903 (floor +0.0023)
  pass  stability: first half +0.3618, second half +0.3408
  pass  block length: S +0.3508 with 56-step blocks; +0.3131 with 29, +0.3795 with 112
  info  recorded causes: without them S would be +0.0086
  info  closure: member detail improves the group forecast by -0.0000
```

The same chain with weaker coupling (`--coupling 0.1 --seed 2`) comes out **Inconclusive**: S = 0.019 is above 0
but not above what independent stores built from the fitted twins produce over three years (0.033). Independent
stores (`twinshare example retail`) come out **No coordination detected**.

The method comes from the Emergence Lab's 90 experiments on fish schools, pigeon flocks, mouse cortex and stock
markets ([Couture 2026](../manuscript/manuscript.md)). twinshare packages it for any panel of time series.

## Install

```bash
uv tool install ./twinshare          # or: pip install ./twinshare
```

Python 3.11+, numpy and matplotlib.

## Try it on data with a known answer

```bash
twinshare example retail --coupling 0.3 --out demo     # stores that respond to the chain's yesterday
twinshare run demo/study.toml                          # -> demo/report/report.html
twinshare example retail --out quiet                   # independent stores: the answer is no coordination
twinshare example retail --leader 0.3 --out flagship   # one store leads: closure and influence find it
```

## Use it on your data

```bash
twinshare init --domain retail       # or generic, markets, neuro: writes study.toml
# edit study.toml: the four declarations, the file and its columns, the recorded causes
twinshare screen study.toml          # quick fit-free preview (the memory gap)
twinshare run study.toml             # full analysis and report
```

The data are a CSV with one row per time step and one column per member (or long format: time, member, value), plus
columns for the recorded causes. Read [PROCEDURE.md](PROCEDURE.md) first: which causes to record (and which would
hide the very interaction you are looking for), how to choose the windows, and how to read each check.

From Python:

```python
from twinshare import Design, analyze, from_arrays
r = analyze(from_arrays(values, causes), Design(windows=(1, 7, 28)))   # values (T, N), causes (T, p)
r.verdict, r.S, r.interval, r.checks
```

## What it computes

| Output | Meaning |
| --- | --- |
| interaction share S | 1 − MSE(group's own law) / MSE(the twin's summed forecast, projected on the law's features); cross-fitted with every stage out of fold |
| memory gap | the share predicted from the members' and the group's autocovariances alone, with nothing fitted |
| trust checks | virtual groups, null twins, power, memory-gap agreement, stability, block length |
| closure | how much individual members' pasts add to the group's own forecast |
| influence | which members' pasts predict the group |

S is **mechanism-blind**: interaction and an unrecorded shared driver look the same. The declarations in the study
file, and in the end an intervention, tell them apart. The report says so every time.

## Validation

* `tests/`: known answers on synthetic chains (independent, coupled, hidden driver, a leader), data loading, the
  command line.
* `validation/lab_replication.py`: with the lab's settings twinshare reproduces its published shares to 1e-15
  (simulated markets; US industry volatility 1936–2026); with its own defaults, the full analysis of US industries
  passes every check (S = 0.021).
* `validation/e70_floors.py`: the virtual-shift correction that building this package found in the lab's screening
  tool.

Safeguards added while generalizing the lab's code, each found by a known-answer test: recorded causes also enter
over the history windows (without that, a promotion yesterday creates S = 0.6 among independent stores); a near-zero
ridge penalty (a penalty of 1 inflates S when a cause varies slowly, like temperature); causes constant on the
training rows are dropped; blocked inner cross-validation for the closure test (GCV assumes independent rows);
virtual offsets over the whole circle; block length capped at ten history windows, with a sensitivity check.

## License

MIT. Cite the manuscript and this package (version, and the SHA-256 printed in each report).
