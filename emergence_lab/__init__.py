"""Emergence Lab: test when a coarse-grained description has its own autonomous dynamics."""

from .closure import (
    ExactClosure,
    closure_gap_bits,
    compression_bits,
    exact_closure,
    macro_determinism,
    predictive_closure,
    robustness,
)
from .info import (
    conditional_mutual_information,
    emergence_psi,
    entropy,
    gaussian_mutual_information,
    mutual_information,
    synergy_report,
)
from .report import EmergenceReport, print_reports
