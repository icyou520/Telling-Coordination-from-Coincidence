"""Build the peer-review manuscript (PDF) from manuscript.md, with the pre-registration record from lab/catalog.py.

    uv run --with markdown python manuscript/build.py     # after whitepaper/build.py (shares its figures)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "whitepaper"))

from build import CHROME, DATE  # noqa: E402
from lab.catalog import EXPERIMENTS, THEMES  # noqa: E402

CSS = """
@page { size: A4; margin: 22mm 22mm 20mm; @bottom-center { content: counter(page);
        font-family: "Helvetica Neue", Arial, sans-serif; font-size: 8.5pt; color: #666; } }
body { font-family: "Charter", Georgia, serif; font-size: 10.5pt; line-height: 1.5; color: #111; margin: 0; }
h1 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 19pt; line-height: 1.2; margin: 0 0 8pt; }
h2, h3, th, figcaption { font-family: "Helvetica Neue", Arial, sans-serif; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; page-break-after: avoid; }
p { margin: 5pt 0; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }
ul, ol { margin: 4pt 0; padding-left: 18pt; }
li { margin: 2pt 0; }
code { font-family: Menlo, monospace; font-size: 8.5pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 8.4pt; width: 100%; }
th { border-bottom: 1pt solid #111; text-align: left; padding: 3pt 5pt; }
td { border-bottom: 0.4pt solid #ccc; padding: 2.5pt 5pt; vertical-align: top; }
tr { page-break-inside: avoid; }
figure { margin: 10pt 0; page-break-inside: avoid; text-align: center; }
figure img { width: 84%; }
figcaption { font-size: 8.6pt; color: #333; text-align: left; margin-top: 4pt; line-height: 1.4; }
"""


def record():
    theme = {k: name for k, name, _ in THEMES}
    rows, n_pass, n_pred, n_scored = [], 0, 0, 0
    for e in EXPERIMENTS:
        m = re.match(r"(\d+)/(\d+)", e["verdict"])
        if m and "trivial" not in e["verdict"] and "hollow" not in e["verdict"]:
            n_pass, n_pred, n_scored = n_pass + int(m[1]), n_pred + int(m[2]), n_scored + 1
        rows.append(f"| {e['id']} | {theme[e['theme']].split('. ', 1)[-1]} | {e['title']} | {e['verdict']} |")
    table = "| # | Theme | Experiment | Verdict |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
    return table, len(EXPERIMENTS), n_scored, n_pass, n_pred


def main():
    table, n_exp, n_scored, n_pass, n_pred = record()
    src = (HERE / "manuscript.md").read_text()
    for k, v in {"{DATE}": DATE, "{N_EXP}": n_exp, "{N_SCORED}": n_scored, "{N_PASS}": n_pass,
                 "{N_PRED}": n_pred, "{PREREG_TABLE}": table}.items():
        src = src.replace(k, str(v))
    src = re.sub(r"(?m)^([^\s\-*\d|].*)\n(?=(?:- |\* |\d+\. ))", r"\1\n\n", src)
    (HERE / "manuscript_rendered.md").write_text(src)
    body = markdown.markdown(src, extensions=["tables", "sane_lists", "md_in_html"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>Manuscript</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path = HERE / "manuscript.html"
    html_path.write_text(html)
    pdf = HERE / "Couture_Interaction_Share_Manuscript.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, capture_output=True)
    html_path.unlink()
    print(pdf, f"({n_exp} experiments, {n_scored} scored, {n_pass}/{n_pred} predictions passed)")


if __name__ == "__main__":
    main()
