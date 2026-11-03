"""The four-property emergence report from the lab's starting hypothesis."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EmergenceReport:
    """Scores for one candidate macro description of one system.

    compression      1 - macro_dim / micro_dim (or 1 - H(Z)/H(X) when discrete)
    autonomy         share of the predictable macro future that the macro layer
                     predicts without micro detail (NaN if nothing is predictable)
    determinism      how strongly the macro state fixes the macro future across
                     different micro realisations (causal consistency)
    robustness       macro trajectory survives micro perturbations
    """

    system: str
    macro: str
    compression: float
    autonomy: float
    determinism: float
    robustness: float | None = None
    extra: dict | None = None

    @property
    def score(self) -> float:
        """Geometric mean of the defined properties (all must be high).

        Undefined (NaN/None) properties are skipped; e.g. autonomy is undefined
        when the macro future is unpredictable, which determinism then penalises.
        """
        vals = [
            v
            for v in (self.compression, self.autonomy, self.determinism, self.robustness)
            if v is not None and not math.isnan(v)
        ]
        prod = 1.0
        for v in vals:
            prod *= min(max(v, 0.0), 1.0)
        return prod ** (1.0 / len(vals))

    HEADER = (
        f"{'system':<14}{'macro description':<34}{'compress':>9}{'autonomy':>9}"
        f"{'determ.':>9}{'robust':>9}{'SCORE':>8}"
    )

    def row(self) -> str:
        cell = lambda v: f"{'n/a':>9}" if v is None or math.isnan(v) else f"{v:9.3f}"
        return (
            f"{self.system:<14}{self.macro:<34}{cell(self.compression)}{cell(self.autonomy)}"
            f"{cell(self.determinism)}{cell(self.robustness)}{self.score:8.3f}"
        )


def print_reports(reports: list[EmergenceReport]) -> str:
    lines = [EmergenceReport.HEADER, "-" * len(EmergenceReport.HEADER)]
    lines += [r.row() for r in reports]
    text = "\n".join(lines)
    print(text)
    return text
