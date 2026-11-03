"""The report: one self-contained HTML file (figures embedded) and a JSON file with every number."""

from __future__ import annotations

import base64
import html
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import Result, noise, null_bar

COLORS = {"pass": "#1a7f37", "warn": "#b35900", "fail": "#c62828", "info": "#555"}
LABELS = {"title": "Question", "members": "Member", "group": "Group", "interaction": "Counts as interaction",
          "recorded_causes": "Recorded causes (used by the twin)", "unrecorded": "Not recorded (could mimic it)"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figures(r: Result) -> list[tuple[str, str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    x, n = r.panel.values, r.panel.shape[1]
    fig, ax = plt.subplots(figsize=(9, 3))
    t = np.arange(len(x))
    for i in range(min(n, 8)):
        ax.plot(t, x[:, i], color="0.75", lw=0.5)
    ax.plot(t, x.mean(1), color="#1f4e8c", lw=1.2, label="group (mean)")
    ticks = np.linspace(0, len(x) - 1, 6).astype(int)
    ax.set_xticks(ticks, [r.panel.times[k] for k in ticks], fontsize=8)
    ax.set_ylabel("member variable")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"The data: {min(n, 8)} of {n} members (grey) and the group", fontsize=10)
    out.append(("data", _png(fig), "What was analysed, after the transform."))

    o, g = r.memory_gap["o"], r.memory_gap["g"]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    k = np.arange(len(o))
    ax.plot(k, o / o[0], "o-", ms=3, color="0.4", label="members (average)")
    ax.plot(k, g / g[0], "o-", ms=3, color="#1f4e8c", label="group")
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("autocorrelation")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Memories, recorded causes removed", fontsize=10)
    out.append(("memory", _png(fig), "If the group remembers its past differently from its members, the group's past "
                "carries information no member's own past does: the memory gap. Equal curves mean no coordination."))

    fig, ax = plt.subplots(figsize=(7, 2.8))
    rows = [("virtual groups", r.virtual, "0.5"), ("independent twins", r.null, "0.5"),
            ("known spillover", [p["S"] for p in r.power], "#b35900")]
    for y, (lab, vals, c) in enumerate(rows):
        ax.scatter(vals, [y] * len(vals), color=c, s=18, zorder=3)
    y = len(rows)
    ax.errorbar([r.S], [y], xerr=[[r.S - r.interval[0]], [r.interval[1] - r.S]], fmt="o", color="#1f4e8c", capsize=3)
    ax.scatter([r.memory_gap["S_pred"]], [y], marker="x", color="#1f4e8c", s=40, zorder=4)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.axvline(null_bar(r), color="#c62828", lw=0.8, ls="--")
    ax.set_yticks(range(y + 1), [lab for lab, _, _ in rows] + ["your data"], fontsize=8)
    ax.set_xlabel("interaction share S")
    ax.set_title("Your share against the reference groups (x: memory-gap prediction; dashed: the bar)", fontsize=10)
    out.append(("share", _png(fig), "Virtual groups and independent twins show what S looks like with no "
                "coordination at this sample size; the known spillover shows what a modest one looks like. Your S, "
                "with its 90% interval, has to clear the dashed bar."))

    fig, ax = plt.subplots(figsize=(7, 2.6))
    s = [p["S"] for p in r.periods]
    ax.bar(range(len(s)), s, color="#1f4e8c")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(s)), [f"{p['start']}\n{p['end']}" for p in r.periods], fontsize=7)
    ax.set_ylabel("S")
    ax.set_title("The share in each sixth of the period", fontsize=10)
    out.append(("periods", _png(fig), "Whether coordination is steady or concentrated in some periods."))

    if r.influence is not None:
        top = np.argsort(-r.influence)[:15]
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(top) + 1))
        ax.barh(range(len(top))[::-1], r.influence[top], color="#1f4e8c")
        ax.set_yticks(range(len(top))[::-1], [r.panel.members[i] for i in top], fontsize=8)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlabel("improvement of the group forecast from this member's history")
        ax.set_title("Influence: whose past predicts the group", fontsize=10)
        out.append(("influence", _png(fig), "Predictive, not causal: a member that leads the group, or that reacts "
                    "first to something unrecorded, both rank high. All near 0 means no member is special."))
    return out


def to_json(r: Result, study: dict | None = None) -> dict:
    return {
        "twinshare": __version__, "verdict": r.verdict, "explanation": r.explanation,
        "S": r.S, "interval_90": list(r.interval), "bar": null_bar(r), "null_spread": noise(r),
        "memory_gap": {k: v for k, v in r.memory_gap.items() if k not in ("o", "g")},
        "memory_gap_floor": r.gap_floor, "virtual": r.virtual, "null": r.null, "power": r.power,
        "periods": r.periods, "halves": list(r.halves), "without_recorded_causes": r.no_covariates,
        "block": r.block, "block_shares": {str(k): v for k, v in r.block_shares.items()},
        "closure": r.closure,
        "influence": None if r.influence is None else dict(zip(r.panel.members, map(float, r.influence))),
        "checks": [c.__dict__ for c in r.checks],
        "data": {"members": r.panel.members, "steps": len(r.panel.times), "first": r.panel.times[0],
                 "last": r.panel.times[-1], "recorded_causes": r.panel.covariate_names, "notes": r.panel.notes,
                 "sha256": r.panel.sources},
        "design": r.design.__dict__ | {"windows": list(r.design.windows)}, "seed": r.seed,
        "study": {k: v for k, v in (study or {}).get("study", {}).items()},
    }


def write(r: Result, out_dir: Path, study: dict | None = None, command: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = to_json(r, study)
    (out_dir / "result.json").write_text(json.dumps(data, indent=2, default=float))
    e = html.escape
    decl = (study or {}).get("study", {})
    undeclared = (study or {}).get("_undeclared", [])
    color = {"Coordination": "#1a7f37", "No coordination": "#444", "Inconclusive": "#b35900",
             "Not interpretable": "#c62828"}
    vcol = next((c for k, c in color.items() if r.verdict.startswith(k)), "#444")
    parts = [f"<h1>{e(decl.get('title') or 'Interaction share report')}</h1>",
             f"<div class='verdict' style='border-color:{vcol}'><div class='v' style='color:{vcol}'>{e(r.verdict)}"
             f"</div><p>{e(r.explanation)}</p></div>"]
    if decl:
        rows = "".join(f"<tr><th>{LABELS[k]}</th><td>{e(str(decl.get(k, '')))}</td></tr>" for k in LABELS
                       if k != "title" and k in decl)
        parts.append(f"<h2>What was asked, and what was assumed</h2><table>{rows}</table>")
    if undeclared:
        parts.append(f"<p class='warn'>Not declared in the study file: {e(', '.join(undeclared))}. A reader cannot "
                     "judge the result without them.</p>")
    gap = r.memory_gap
    parts.append("<h2>Numbers</h2><table>"
                 f"<tr><th>Interaction share S</th><td>{r.S:+.4f} (90% interval {r.interval[0]:+.4f} to "
                 f"{r.interval[1]:+.4f})</td></tr>"
                 f"<tr><th>Bar from independent twins</th><td>{null_bar(r):+.4f}</td></tr>"
                 f"<tr><th>Memory-gap prediction (no fitting)</th><td>{gap['S_pred']:+.4f} (virtual floor "
                 f"{r.gap_floor:+.4f}); member memory {gap['rho_member']:.3f}, group memory {gap['rho_group']:.3f}"
                 "</td></tr>"
                 + (f"<tr><th>Without the recorded causes</th><td>{r.no_covariates:+.4f}</td></tr>"
                    if r.no_covariates is not None else "")
                 + (f"<tr><th>Closure (member detail gain)</th><td>{r.closure:+.4f}</td></tr>"
                    if r.closure is not None else "") + "</table>")
    rows = "".join(f"<tr><td>{e(c.name)}</td><td style='color:{COLORS[c.status]};font-weight:600'>{c.status}</td>"
                   f"<td>{e(c.value)}</td><td>{e(c.meaning)}</td></tr>" for c in r.checks)
    parts.append(f"<h2>Trust checks</h2><table class='checks'><tr><th>check</th><th></th><th>result</th>"
                 f"<th>what it means</th></tr>{rows}</table>")
    for name, png, caption in figures(r):
        parts.append(f"<figure><img src='data:image/png;base64,{png}' alt='{name}'><figcaption>{e(caption)}"
                     "</figcaption></figure>")
    p = r.panel
    notes = "".join(f"<li>{e(n)}</li>" for n in p.notes) or "<li>none</li>"
    src = "".join(f"<li><code>{e(f)}</code> sha256 <code>{h}</code></li>" for f, h in p.sources.items())
    d = r.design
    parts.append(
        "<h2>Data and settings</h2>"
        f"<p>{p.shape[1]} members, {p.shape[0]} time steps ({e(p.times[0])} to {e(p.times[-1])}). Recorded causes: "
        f"{e(', '.join(p.covariate_names) or 'none')}.</p><p>Changes made to the data:</p><ul>{notes}</ul>"
        f"<p>Horizon {d.horizon}, history windows {list(d.windows)}, {d.folds} folds of "
        f"{r.block}-step blocks, purged: {d.purge}, causes over the history windows: "
        f"{d.lagged_causes}, penalty {d.lam:g}. Seed {r.seed}.</p>"
        "<h2>Reproduce</h2>"
        f"<ul>{src}</ul><p>twinshare {__version__}, Python {platform.python_version()}, numpy {np.__version__}"
        f"{', command: <code>' + e(command) + '</code>' if command else ''}.</p>"
        "<h2>How to read this</h2><p>The twin keeps every member and switches interaction off: each member is "
        "forecast from its own past and the recorded causes only. S is the share of the group forecast's error "
        "under the twin that the group's own past removes. It is mechanism-blind: interaction and an unrecorded "
        "shared driver that each member reveals poorly look the same. A positive S says the members are "
        "coordinated beyond their own histories and the recorded causes; whether that is interaction is decided "
        "by the declarations above, and in the end by an intervention.</p>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(decl.get('title', 'twinshare report'))}</title>
<style>body{{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:960px;margin:2em auto;
padding:0 1em;color:#222}}h1{{font-size:1.5em}}h2{{font-size:1.15em;margin-top:1.8em;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;font-size:.92em}}th,td{{text-align:left;vertical-align:top;
padding:.35em .6em;border-bottom:1px solid #eee}}th{{width:24%;color:#444}}.checks th{{width:auto}}
.verdict{{border-left:5px solid;padding:.4em 1em;background:#fafafa}}.v{{font-size:1.3em;font-weight:700}}
figure{{margin:1.5em 0}}img{{max-width:100%}}figcaption{{font-size:.88em;color:#555}}.warn{{color:#b35900}}
code{{font-size:.85em}}</style></head><body>{''.join(parts)}</body></html>"""
    path = out_dir / "report.html"
    path.write_text(page)
    return path
