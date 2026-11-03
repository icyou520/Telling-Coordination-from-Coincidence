"""Build the release: dist/emergence-lab-<version>/ plus two zips (code and papers; results).

    uv run python make_release.py 1.0.0

Only the paths listed below are copied: no third-party data (see DATA.md), simulation caches or working notes.
SHA256SUMS files list every file: one at the top of the code folder, one in results/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["README.md", "DATA.md", "FINDINGS.md", "LICENSE", "LICENSE-CC-BY-4.0.md", "CITATION.cff", "pyproject.toml",
         "uv.lock", ".python-version", "run.py", "explorer.py", "verify_preregistration.py", "make_release.py",
         "results/e39_preregistration.txt", "data/stocks/constituents.json", "data/stocks_fresh/constituents.json",
         "data/intervention/vr_urls.txt", "manuscript/manuscript.md", "manuscript/build.py",
         "manuscript/Couture_Interaction_Share_Manuscript.pdf", "whitepaper/whitepaper.md", "whitepaper/build.py",
         "whitepaper/Couture_Interaction_Share_White_Paper.pdf"]
DIRS = ["experiments", "emergence_lab", "lab", "tests", "lean", "whitepaper/figures", "twinshare"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "site", ".venv", "report"}
SKIP_FILES = {".DS_Store"}
RESULT_SUFFIXES = {".json", ".png", ".txt", ".log"}


def walk(d: Path):
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name not in SKIP_FILES and not SKIP_DIRS & set(p.relative_to(ROOT).parts):
            yield p


def copy(paths, dest: Path) -> int:
    n = 0
    for p in paths:
        out = dest / p.relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def sums(folder: Path) -> None:
    lines = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(folder).as_posix()}")
    (folder / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def pack(folder: Path, zip_path: Path, top: bool) -> None:
    """Zip folder; with top=False the entries start inside it (the results zip unpacks to results/ in place)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                rel = p.relative_to(folder)
                z.write(p, Path(folder.name) / rel if top else rel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    args = ap.parse_args()
    code = args.out / f"emergence-lab-{args.version}"
    results = args.out / f"emergence-lab-results-{args.version}"
    for target in (code, results):
        if target.exists():
            raise SystemExit(f"{target} exists; remove it or choose another version")

    paths = [ROOT / f for f in FILES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    n = copy(paths + [p for d in DIRS for p in walk(ROOT / d)], code)
    sums(code)
    m = copy([p for p in walk(ROOT / "results") if p.suffix in RESULT_SUFFIXES
              and p.name != "e39_preregistration.txt"], results)
    sums(results / "results")
    for folder, top in ((code, True), (results, False)):
        zip_path = folder.parent / f"{folder.name}.zip"
        pack(folder, zip_path, top)
        print(f"{zip_path}  {zip_path.stat().st_size / 1e6:.1f} MB")
    print(f"{n} files in {code.name}, {m} in {results.name}")


if __name__ == "__main__":
    main()
