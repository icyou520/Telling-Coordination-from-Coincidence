"""Command line.

    twinshare init --domain retail            write study.toml to fill in
    twinshare run study.toml                  full analysis -> report/report.html and report/result.json
    twinshare screen study.toml               memory gap only (seconds, nothing fitted)
    twinshare example retail --out demo       a synthetic chain with a known answer, plus its study file
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import analyze
from .data import load
from .engine import memory_gap, memories, residualize, s_pred, shift
from .examples import write_retail
from .report import write
from .study import DOMAINS, design, load_study, template


def _init(a):
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f"{out} exists (use --force to overwrite)")
    out.write_text(template(a.domain))
    print(f"wrote {out}: fill in [study] and [data], then run  twinshare run {out}")


def _load(path):
    study = load_study(Path(path))
    panel = load(study, Path(path).parent)
    return study, panel


def _run(a):
    study, panel = _load(a.study)
    d = design(study)
    seed = int(study.get("run", {}).get("seed", 0))
    print(f"{panel.shape[1]} members, {panel.shape[0]} steps, recorded causes: {', '.join(panel.covariate_names) or 'none'}")
    for n in panel.notes:
        print(f"  note: {n}")
    if study["_undeclared"]:
        print(f"  warning: [study] does not declare {', '.join(study['_undeclared'])}")
    r = analyze(panel, d, seed, n_null=4 if a.quick else 8, detail=not a.quick,
                progress=lambda m: print(f"  ... {m}", flush=True))
    out = Path(a.out) if a.out else Path(a.study).parent / "report"
    path = write(r, out, study, "twinshare " + " ".join(shlex.quote(s) for s in sys.argv[1:]))
    print(f"\n{r.verdict}\n{r.explanation}\n")
    for c in r.checks:
        print(f"  {c.status:5s} {c.name}: {c.value}")
    print(f"\nreport: {path.resolve()}")


def _screen(a):
    study, panel = _load(a.study)
    d = design(study)
    gap = memory_gap(panel.values, panel.covariates, d)
    u = residualize(panel.values, panel.covariates)
    rng = np.random.default_rng(int(study.get("run", {}).get("seed", 0)))
    floor = float(np.median([s_pred(*memories(shift(u, rng), d.lead), d) for _ in range(5)]))
    print(f"{panel.shape[1]} members, {panel.shape[0]} steps")
    print(f"memory gap S_pred {gap['S_pred']:+.4f}  (virtual floor {floor:+.4f}; near or below it the screen is silent)")
    print(f"member memory {gap['rho_member']:.3f}, group memory {gap['rho_group']:.3f}")
    print("This is a screen: run  twinshare run  for the measured share and the trust checks.")


def _example(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_retail(out / "sales.csv", seed=a.seed, coupling=a.coupling, hidden=a.hidden, leader=a.leader)
    (out / "study.toml").write_text(template("retail"))
    truth = ("independent stores: the answer is no coordination" if not (a.coupling or a.hidden or a.leader) else
             ", ".join(f"{k} {v}" for k, v in (("coupling", a.coupling), ("hidden driver", a.hidden),
                                               ("leader", a.leader)) if v))
    print(f"wrote {out}/sales.csv and {out}/study.toml ({truth})\nrun:  twinshare run {out}/study.toml")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twinshare", description="Interaction share: is a group coordinated beyond "
                                 "its members' own histories and the recorded causes?")
    ap.add_argument("--version", action="version", version=f"twinshare {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write a study file template")
    p.add_argument("--domain", default="generic", choices=sorted(DOMAINS))
    p.add_argument("--out", default="study.toml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=_init)
    p = sub.add_parser("run", help="full analysis and report")
    p.add_argument("study")
    p.add_argument("--out", help="report folder (default: report/ next to the study file)")
    p.add_argument("--quick", action="store_true", help="fewer simulations, no closure or influence")
    p.set_defaults(fn=_run)
    p = sub.add_parser("screen", help="memory gap only")
    p.add_argument("study")
    p.set_defaults(fn=_screen)
    p = sub.add_parser("example", help="write a synthetic data set with a known answer")
    p.add_argument("kind", choices=["retail"])
    p.add_argument("--out", default="example")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coupling", type=float, default=0.0, help="stores respond to the chain's yesterday")
    p.add_argument("--hidden", type=float, default=0.0, help="an unrecorded regional driver")
    p.add_argument("--leader", type=float, default=0.0, help="stores follow store_01's yesterday")
    p.set_defaults(fn=_example)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (ValueError, FileNotFoundError, KeyError) as err:
        sys.exit(f"error: {err}")


if __name__ == "__main__":
    main()
