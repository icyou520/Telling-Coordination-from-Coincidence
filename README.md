# Emergence Lab: telling coordination from coincidence

Code, pre-registration record and papers for *Telling coordination from coincidence: a no-interaction twin for
collective systems* (E. J. Couture).

A group can look coordinated because its members interact, or because each member follows the same environment and
its own habits. The lab separates the two by fitting a **no-interaction twin**: the same members with every coupling
switched off. The **interaction share** S = 1 − MSE(group)/MSE(twin) is the part of the group's predictability that only
interaction can explain. It was tested in simulations with a known answer and then on schooling fish, zebrafish,
pigeon flocks, mouse cortex (including a causal silencing experiment) and stock markets, and then pushed to its
extremes: the Ising phase transition, groups of 2 to 10,000, time reversal, a flawed random-number generator (RANDU)
and the prime numbers.

90 experiments (e0–e89), 73 of them scored against predictions written and hashed before the run: 342 of 492
predictions passed. Every experiment, including the failures, is written up in [FINDINGS.md](FINDINGS.md).

**Papers:** [manuscript (PDF)](manuscript/Couture_Interaction_Share_Manuscript.pdf) ·
[white paper (PDF)](whitepaper/Couture_Interaction_Share_White_Paper.pdf).
**Results** (raw JSON and figures for every experiment):
[`emergence-lab-results-1.0.1.zip`](https://github.com/icyou520/Telling-Coordination-from-Coincidence/releases)
on the Releases page.

## Contents

| Path | Contents |
| --- | --- |
| `manuscript/` | Peer-review manuscript (`manuscript.md`, PDF, `build.py`) |
| `whitepaper/` | White paper v2.2 (`whitepaper.md`, PDF, figures, `build.py`) |
| `FINDINGS.md` | Every experiment: predictions, results, revisions, and the command that reproduces it |
| `results/e39_preregistration.txt` | Pre-registration log: SHA-256 of each experiment's code (and fresh data) before it ran |
| `verify_preregistration.py` | Checks the files against the log |
| `experiments/` | One file per experiment (`eNN_*.py`, docstring = pre-registered predictions), its figures, data loaders |
| `emergence_lab/` | Library: closure tests, information measures, layers, simple systems, matplotlib explorer |
| `lab/` | Visual lab: a self-contained web page with concepts, live simulations and every experiment's charts |
| `twinshare/` | The method as a standalone package for any panel of time series (own README, PROCEDURE and tests) |
| `lean/` | Lean 4 proofs: a macro law exists iff the coarse-graining is closed; interaction share identities |
| `tests/` | Test suite |
| `run.py` | Runs experiments by name |

Experiment results (raw JSON and figures, about 95 MB) are distributed as a separate archive,
`emergence-lab-results-<version>.zip`, attached to each
[GitHub release](https://github.com/icyou520/Telling-Coordination-from-Coincidence/releases); unzip it in this folder
to get `results/`. In that archive, `results/SHA256SUMS` lists a checksum for every file:
`cd results && shasum -a 256 -c SHA256SUMS`.

## Install

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically from `.python-version`).

```bash
git clone https://github.com/icyou520/Telling-Coordination-from-Coincidence.git
cd Telling-Coordination-from-Coincidence
uv sync
uv run pytest -q
```

## Reproduce

```bash
uv run python run.py --help              # every experiment name
uv run python run.py toy ca flock        # run some (simulation-only ones need no data)
uv run python run.py --quick toy         # smaller, faster simulations
```

Each FINDINGS section ends with its command (for example `uv run python run.py lightdose` for e83). Experiments on real
data need the public data sets in `data/`; [DATA.md](DATA.md) lists where each comes from and where it goes. Large
runs use all CPU cores and can take hours; the flock experiments cache simulations in `results/`.

## Use the method on your own data

`twinshare/` packages the interaction share for anyone with a panel of time series (stores, sectors, neurons,
sensors): a study file for the domain assumptions, one command, an HTML report with trust checks. See
[twinshare/README.md](twinshare/README.md) and [twinshare/PROCEDURE.md](twinshare/PROCEDURE.md).

```bash
cd twinshare && uv run twinshare example retail --coupling 0.3 --out demo && uv run twinshare run demo/study.toml
```

## Check the pre-registration

Before each experiment ran, the SHA-256 hashes of its code (and, for freshly downloaded data, of the data) were
appended to `results/e39_preregistration.txt` with a UTC timestamp. Later edits are logged with a note and a new hash.

```bash
uv run python verify_preregistration.py          # every file against its last recorded hash
uv run python verify_preregistration.py --all
```

Data files show as missing until downloaded. The log is append-only text; it is evidence of order, not a third-party
timestamp.

## Papers

```bash
MPLCONFIGDIR=/tmp/mpl uv run --with markdown python whitepaper/build.py
MPLCONFIGDIR=/tmp/mpl uv run --with markdown python manuscript/build.py
```

Both render markdown to PDF with headless Google Chrome (path set in `whitepaper/build.py`); the white paper's figures
are drawn from `results/`.

## Visual lab and explorer

```bash
uv run python -m lab.build          # then open lab/site/index.html
uv run python explorer.py           # matplotlib explorer of layers and closure
```

## Lean proofs

Core Lean 4, no Mathlib: `lean lean/Emergence.lean` and `lean lean/InteractionShare.lean`.

## License

Code: MIT ([LICENSE](LICENSE)). Papers, write-ups and figures (`manuscript/`, `whitepaper/`, `FINDINGS.md`,
`results/` figures): CC BY 4.0 ([LICENSE-CC-BY-4.0.md](LICENSE-CC-BY-4.0.md)). Third-party data are not included and
keep their own licenses (see DATA.md). Cite with [CITATION.cff](CITATION.cff).
