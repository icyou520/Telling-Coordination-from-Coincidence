from pathlib import Path

from lab.build import findings_sections, markdown
from lab.catalog import EXPERIMENTS, THEMES

RESULTS = Path(__file__).resolve().parent.parent / "results"


def test_catalog_is_consistent():
    themes = {t for t, _, _ in THEMES}
    ids = [e["id"] for e in EXPERIMENTS]
    assert len(ids) == len(set(ids))
    for e in EXPERIMENTS:
        assert e["theme"] in themes
        assert e["tone"] in {"pass", "partial", "fail", "explore"}
        for f in e["figures"]:
            assert f["what"] and f["read"] and f["tells"]
            if any(RESULTS.glob("*.json")):
                assert (RESULTS / f["path"]).exists(), f["path"]


def test_every_experiment_has_a_writeup():
    subs, _ = findings_sections()
    for e in EXPERIMENTS:
        assert any(h.startswith(e["section"]) or e["section"] in h for h in subs), e["id"]


def test_markdown_subset():
    out = markdown("# T\n\n| a | b |\n|---|---|\n| **x** | `y` |\n\n* one\n  * two\n")
    assert "<h2>T</h2>" in out
    assert "<td><strong>x</strong></td><td><code>y</code></td>" in out
    assert out.count("<ul>") == 2
