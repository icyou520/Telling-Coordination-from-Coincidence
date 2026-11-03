"""Launch the interactive Emergence Lab explorer.

    uv run python explorer.py                     # open the window
    uv run python explorer.py --snapshot results  # save PNGs of every view instead
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, help="render every system and view to PNGs in this folder")
    args = parser.parse_args()

    if args.snapshot is None:
        from emergence_lab.viz.app import main as run

        run()
        return

    matplotlib.use("Agg")
    from emergence_lab.viz.app import EXPLORE, TESTS, Explorer

    args.snapshot.mkdir(parents=True, exist_ok=True)
    app = Explorer(interactive=False)
    for name in app.views:
        app.select(system=name, mode=EXPLORE)
        for _ in range(80):
            app._on_step(None)
        slug = name.lower().replace(" ", "_")
        app.fig.savefig(args.snapshot / f"explorer_{slug}_layers.png", dpi=90)
        app.select(mode=TESTS)
        app.fig.savefig(args.snapshot / f"explorer_{slug}_tests.png", dpi=90)
        print(f"saved {name}")


if __name__ == "__main__":
    main()
