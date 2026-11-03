"""Check files against the SHA-256 hashes in the pre-registration log (results/e39_preregistration.txt).

    uv run python verify_preregistration.py            # summary plus every file that differs
    uv run python verify_preregistration.py --all      # one line per file

Each file is compared with the last hash recorded for it (files re-hashed after a documented revision keep their
history in the log; the note next to each entry says why). Data files are reported as missing until the data are
downloaded (see DATA.md).
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "results" / "e39_preregistration.txt"
LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
CONCAT = " (concatenated, sorted by name)"


def sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def files(entry: str) -> list[Path]:
    """The file(s) behind a log entry; an entry 'dir/* (concatenated, sorted by name)' is every file in dir."""
    if entry.endswith(CONCAT):
        d = ROOT / entry.removesuffix(CONCAT).removesuffix("/*")
        return sorted(p for p in d.iterdir() if p.is_file()) if d.is_dir() else []
    p = ROOT / entry
    return [p] if p.exists() else []


def recorded() -> dict[str, list[str]]:
    """Every hash recorded for each path, in log order."""
    out: dict[str, list[str]] = {}
    for line in LOG.read_text().splitlines():
        m = LINE.match(line.strip())
        if m:
            out.setdefault(m.group(2), []).append(m.group(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="list every file, not only the ones that differ")
    args = ap.parse_args()
    counts = {"match": 0, "earlier": 0, "differs": 0, "missing": 0}
    for path, hashes in recorded().items():
        ps = files(path)
        if not ps:
            status = "missing"
        else:
            now = sha256(ps)
            status = "match" if now == hashes[-1] else "earlier" if now in hashes else "differs"
        counts[status] += 1
        if args.all or status in ("earlier", "differs"):
            print(f"{status:8s} {path}")
    print(f"\n{sum(counts.values())} files in the log: {counts['match']} match their last recorded hash, "
          f"{counts['earlier']} match an earlier one, {counts['differs']} differ, "
          f"{counts['missing']} missing (data not downloaded, or not part of this release)")


if __name__ == "__main__":
    main()
