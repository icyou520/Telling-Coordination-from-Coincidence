"""Build the visual Emergence Lab into lab/site/.

    uv run python -m lab.build          # then open lab/site/index.html

The site is static and self-contained (no server, no network): figures are
copied from results/, the write-ups come from FINDINGS.md, the explainers
from lab/catalog.py, and real fish clips are exported from data/puy2024/
when that data is present.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import numpy as np

from lab.catalog import EXPERIMENTS, THEMES

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "lab"
SITE = LAB / "site"
RESULTS = ROOT / "results"

FISH_CLIPS = ((8, 0, 300.0, 300.0), (39, 0, 300.0, 180.0))  # n, rec, start s, length s
FISH_HZ = 10


# ---------------------------------------------------------------- markdown

def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = s.replace("\\|", "|").replace("\\(", "").replace("\\)", "")
    return s


def markdown(md: str) -> str:
    """A small Markdown subset: headings, paragraphs, nested lists, tables, code, display math."""
    out: list[str] = []
    lines = md.splitlines()
    i, para = 0, []

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        if s.startswith("\\["):
            flush()
            j = i
            while "\\]" not in lines[j]:
                j += 1
            body = " ".join(x.strip() for x in lines[i:j + 1]).replace("\\[", "").replace("\\]", "")
            out.append('<div class="math">' + html.escape(body) + "</div>")
            i = j + 1
            continue
        m = re.match(r"(#{1,6})\s+(.*)", s)
        if m:
            flush()
            lvl = min(len(m.group(1)) + 1, 6)
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{2,}", lines[i + 1].strip()):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in re.split(r"(?<!\\)\|", r.strip("|"))] for r in rows]
            head, body = cells[0], cells[2:]
            t = "<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append('<div class="table-wrap">' + t + "</tbody></table></div>")
            continue
        if re.match(r"^\s*([*-]|\d+\.)\s+", line):
            flush()
            items = []
            while i < len(lines) and (re.match(r"^\s*([*-]|\d+\.)\s+", lines[i])
                                      or (lines[i].startswith("  ") and lines[i].strip() and items)):
                ln = lines[i]
                m2 = re.match(r"^(\s*)([*-]|\d+\.)\s+(.*)", ln)
                if m2:
                    items.append([len(m2.group(1)), m2.group(2)[0].isdigit(), m2.group(3)])
                else:
                    items[-1][2] += " " + ln.strip()
                i += 1
            out.append(_list_html(items))
            continue
        if s.startswith(">"):
            flush()
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(q)) + "</blockquote>")
            continue
        if not s or s == "---":
            flush()
            i += 1
            continue
        para.append(s)
        i += 1
    flush()
    return "\n".join(out)


def _list_html(items) -> str:
    out, stack = [], []
    for indent, ordered, text in items:
        while stack and indent < stack[-1][0]:
            out.append("</li></" + stack.pop()[1] + ">")
        if not stack or indent > stack[-1][0]:
            tag = "ol" if ordered else "ul"
            stack.append((indent, tag))
            out.append(f"<{tag}><li>")
        else:
            out.append("</li><li>")
        out.append(_inline(text))
    while stack:
        out.append("</li></" + stack.pop()[1] + ">")
    return "".join(out)


# ---------------------------------------------------------------- FINDINGS

def findings_sections() -> tuple[dict[str, str], dict[str, str]]:
    """(### experiment sections, ## top-level sections), Markdown bodies keyed by heading."""
    text = (ROOT / "FINDINGS.md").read_text()
    subs, tops = {}, {}
    cur_top, cur_sub, buf_top, buf_sub = None, None, [], []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur_sub:
                subs[cur_sub] = "\n".join(buf_sub)
            if cur_top:
                tops[cur_top] = "\n".join(buf_top)
            cur_top, cur_sub, buf_top, buf_sub = line[3:].strip(), None, [], []
            continue
        if line.startswith("### "):
            if cur_sub:
                subs[cur_sub] = "\n".join(buf_sub)
            cur_sub, buf_sub = line[4:].strip(), []
            continue
        (buf_sub if cur_sub else buf_top).append(line)
    if cur_sub:
        subs[cur_sub] = "\n".join(buf_sub)
    if cur_top:
        tops[cur_top] = "\n".join(buf_top)
    return subs, tops


def _section(subs: dict[str, str], key: str) -> tuple[str, str]:
    for head, body in subs.items():
        if head.startswith(key) or key in head:
            return head, body
    return "", ""


# ---------------------------------------------------------------- fish clips

def fish_clips() -> dict:
    try:
        from experiments.e31_real_fish import virtual
        from experiments.fish_data import FPS, load
    except Exception:
        return {}
    clips = {}
    for n, rec, start, length in FISH_CLIPS:
        try:
            pos = load(n, rec)
        except (FileNotFoundError, OSError):
            continue
        step = FPS // FISH_HZ
        a, b = int(start * FPS), int((start + length) * FPS)
        for kind, data in (("real", pos), ("virtual", virtual(pos))):
            p = data[a:b:step].copy()
            for f in range(n):
                for c in range(2):
                    col = p[:, f, c]
                    bad = np.isnan(col)
                    if bad.all():
                        col[:] = 50.0
                    elif bad.any():
                        idx = np.where(~bad, np.arange(len(col)), 0)
                        np.maximum.accumulate(idx, out=idx)
                        col[:] = col[idx]
                        first = np.argmax(~bad)
                        col[:first] = col[first]
            q = np.round(p * 10).astype(int)
            clips[f"{n}_{kind}"] = {"n": n, "rec": rec, "hz": FISH_HZ, "start": start,
                                    "frames": len(q), "xy": q.reshape(-1).tolist()}
    return clips


# ---------------------------------------------------------------- build

def build() -> Path:
    SITE.mkdir(parents=True, exist_ok=True)
    for old in SITE.iterdir():  # keep the folder itself, so a running local server survives
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    (SITE / "figures").mkdir()
    for f in (LAB / "static").iterdir():
        shutil.copy2(f, SITE / f.name)

    subs, tops = findings_sections()
    experiments, n_figs = [], 0
    for e in EXPERIMENTS:
        head, body = _section(subs, e["section"])
        figs = []
        for f in e["figures"]:
            src = RESULTS / f["path"]
            if not src.exists():
                print(f"  missing figure: {f['path']}")
                continue
            dst = SITE / "figures" / f["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            figs.append({**f, "src": "figures/" + f["path"]})
            n_figs += 1
        experiments.append({k: v for k, v in e.items() if k not in ("figures", "section")}
                           | {"figures": figs, "writeup_title": head, "writeup": markdown(body)})

    for name in ("fig1_twin.png", "fig2_walkers.png", "fig3_transitions.png", "fig4_fish.png"):
        src = ROOT / "whitepaper" / "figures" / name
        if src.exists():
            shutil.copy2(src, SITE / "figures" / name)
    video = ROOT / "video" / "interaction_share_explainer.mp4"
    if video.exists():
        shutil.copy2(video, SITE / video.name)

    pages = {}
    for key, title in (("1. Summary", "summary"), ("2. The emerging picture", "picture"),
                       ("3. Measures that survived", "measures"), ("5. Equations", "equations"),
                       ("6. Lessons for the theory", "lessons"), ("7. Open problems", "open")):
        for head, body in tops.items():
            if head.startswith(key):
                pages[title] = {"title": head, "html": markdown(body)}

    clips = fish_clips()
    data = {"themes": [{"id": t, "title": a, "question": q} for t, a, q in THEMES],
            "experiments": experiments, "pages": pages,
            "video": video.exists(), "fish": sorted(clips)}
    (SITE / "data.js").write_text("window.LAB = " + json.dumps(data, ensure_ascii=False) + ";\n")
    (SITE / "fish.js").write_text("window.FISH = " + json.dumps(clips, separators=(",", ":")) + ";\n")
    print(f"Emergence Lab: {len(experiments)} experiments, {n_figs} figures, "
          f"{len(clips)} fish clips -> {SITE / 'index.html'}")
    return SITE / "index.html"


if __name__ == "__main__":
    build()
