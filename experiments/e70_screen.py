"""Experiment 70 (exploratory, no predictions): the memory-gap tool as a screen across every dataset in the lab.

The tool (experiments/memgap.py) needs no twin, so it also runs where the full method could not: pigeon flocks
(e35-e37, no twin reproduced cohesion) and stocks (e66, virtual groups broken by the short series; the tool's
own baseline shifts members within the series, but the same caveat applies). Everything here is exploratory and
reported as such; numbers are screens, not results.

  fish        adult fish (Puy) and zebrafish (Heras): heading vectors, 1 s ahead (e68 windows).
  pigeons     flocks A, B, C (Nagy): heading vectors on the 0.2 s GPS samples where all birds fly, contiguous
              segments only, 1 s ahead (5 samples).
  ALM         Inagaki sessions, all control trials, the 1.5 s before the go cue (bins 50-199 of 10 ms), counts
              with cross-fitted side-specific means removed (e67 residualize), next 50 ms (e62 windows). Checked
              against e67's cross-fitted measured share on the same trials (alm_check).
  markets     49 industries 1972-2026 and 40 industries 1936-1971, log volatility, 1 day ahead (HAR windows);
              S&P 500 stocks 2013-2018, log daily range.
Brain areas (Steinmetz) are e67's R2 and not repeated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.e62_neural_share_repaired import H, HIST
from experiments.e67_memory_gap import HAR_WINDOWS
from experiments.e68_fish_memory_gap import WINDOWS as FISH_WINDOWS
from experiments.memgap import screen

PIGEON_WINDOWS = ((5, 6), (6, 9), (9, 13))


def unit(theta):
    return np.stack([np.cos(theta), np.sin(theta)], -1)


def fish():
    from experiments import e38_zebrafish as z
    from experiments.e31_real_fish import kinematics
    from experiments.fish_data import RECORDINGS, load
    out = []
    for n, recs in RECORDINGS.items():
        for r in recs:
            th = kinematics(load(n, r))[1]
            th = th[np.isfinite(th).all(1)]
            out.append({"domain": "fish", "name": f"adult N{n} rec{r}", **screen([unit(th)], FISH_WINDOWS)})
    for n in z.SIZES:
        for r in z.RECS:
            pos, _, dt = z.load(n, r)
            out.append({"domain": "fish", "name": f"zebrafish N{n} rec{r}",
                        **screen([unit(z.kinematics(pos, dt)[1])], FISH_WINDOWS)})
    return out


def pigeons():
    from experiments.pigeon_data import FLOCKS, flights, load
    out = []
    for flock in FLOCKS:
        segs = []
        for fl in flights(flock):
            s = load(fl, 2)
            starts = np.flatnonzero(~s["cont"])
            for a, b in zip(starts, list(starts[1:]) + [len(s["cont"])]):
                if b - a > 60:
                    segs.append(unit(s["theta"][a:b]))
        out.append({"domain": "pigeons", "name": f"flock {flock}", **screen(segs, PIGEON_WINDOWS)})
    return out


def alm():
    from experiments import inagaki_data as idata
    from experiments.e64_input_silencing import eligible
    from experiments.e67_memory_gap import evaluate, residualize, trial_design
    out = []
    for name in eligible():
        s = idata.load(name)
        c = ~s["silenced"]
        x = s["x"][:, c, 50:200].astype(float)
        fold = np.random.default_rng(0).permutation(np.arange(x.shape[1]) % 5)
        x = residualize(x, s["left"][c].astype(int), fold)
        segs = [x[:, t].T for t in range(x.shape[1])]
        measured = evaluate(x, *trial_design()[:3], np.arange(49, x.shape[-1] - H, 2), fold)["S"]
        out.append({"domain": "ALM", "name": name, "S_measured": measured,
                    **screen(segs, HIST, target=range(1, H + 1), baseline="permute")})
    return out


def markets():
    from experiments import finance_data as fd
    out = []
    for name, (title, start, end) in {"industries 1972-2026": ("Average Value Weighted Returns", 19720101, 99999999),
                                       "industries 1936-1971": ("Average Value Weighted Returns", 19360101, 19711231)}.items():
        _, r, _ = fd.industries(title, start, end)
        out.append({"domain": "markets", "name": name, **screen([np.log(np.abs(r) + 0.05)], HAR_WINDOWS)})
    _, x, _ = fd.stocks()
    out.append({"domain": "markets", "name": "S&P 500 stocks 2013-2018", **screen([np.log(x + 0.05)], HAR_WINDOWS)})
    return out


def analyze(rows) -> str:
    lines = ["Experiment 70 (exploratory): memory-gap screen across the lab's datasets", "",
             "  domain   dataset                        members  S_pred   virtual  member mem  group mem"]
    for r in rows:
        lines.append(f"  {r['domain']:8} {r['name']:30} {r['members']:5d}  {r['S_pred']:+.3f}   {r['virtual']:+.3f}   "
                     f"{r['rho_m']:.3f}       {r['rho_g']:.3f}")
    alm = [r for r in rows if "S_measured" in r]
    if alm:
        from scipy.stats import spearmanr
        m, p = np.array([r["S_measured"] for r in alm]), np.array([r["S_pred"] for r in alm])
        lines += ["", f"  ALM check against the cross-fitted measured share (e67 method, same trials): median S "
                      f"{np.median(m):+.3f}, median S_pred {np.median(p):+.3f}, Spearman {spearmanr(m, p)[0]:.2f}, "
                      f"S > 0 in {(m > 0).sum()}/{len(m)}"]
    lines += ["", "  Medians by domain:"]
    for d in dict.fromkeys(r["domain"] for r in rows):
        v = [r["S_pred"] for r in rows if r["domain"] == d]
        lines.append(f"  {d:8} {np.median(v):+.3f} ({len(v)} datasets)")
    text = "\n".join(lines)
    print(text)
    return text


def main(out_dir: Path | None = None, quick: bool = False) -> str:
    out_dir = out_dir or Path("results")
    rows = fish() + pigeons() + alm() + markets()
    (out_dir / "e70_screen_raw.json").write_text(json.dumps(rows))
    text = analyze(rows)
    from experiments.e70_figures import make_figures
    make_figures(rows, out_dir / "e70_figures", text)
    return text


if __name__ == "__main__":
    main()
