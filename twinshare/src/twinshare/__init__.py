"""twinshare: is a group coordinated beyond its members' own histories and the shared causes you recorded?

    from twinshare import from_arrays, analyze
    result = analyze(from_arrays(values, covariates))      # values (T, N), covariates (T, p)
    print(result.verdict, result.S, result.interval)
"""

__version__ = "0.1.0"

from .analysis import Result, analyze  # noqa: E402
from .data import Panel, from_arrays, load  # noqa: E402
from .engine import Design  # noqa: E402

__all__ = ["Design", "Panel", "Result", "analyze", "from_arrays", "load", "__version__"]
