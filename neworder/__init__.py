import importlib.metadata
import warnings
from types import ModuleType

__version__ = importlib.metadata.version("neworder")

from _neworder_core import (  # ty:ignore[unresolved-import]
    LinearTimeline,
    Model,
    MonteCarlo,
    NoTimeline,
    NumericTimeline,
    SplitMix64,
    Timeline,
    checked,
    df,
    freethreaded,
    log,
    mpi,
    run,
    thread_id,
    time,
    verbose,
)

from .domain import Domain, Edge, Space, StateGrid
from .mc import as_np
from .timeline import CalendarTimeline

__all__: list[str] = [
    "CalendarTimeline",
    "Domain",
    "Edge",
    "LinearTimeline",
    "Model",
    "MonteCarlo",
    "NoTimeline",
    "NumericTimeline",
    "Space",
    "SplitMix64",
    "StateGrid",
    "Timeline",
    "as_np",
    "checked",
    "df",
    "freethreaded",
    "log",
    "mpi",
    "run",
    "stats",
    "thread_id",
    "time",
    "verbose",
]


# stats is served lazily so that accessing it (including via `from neworder import stats`) raises a warning
def __getattr__(name: str) -> ModuleType:
    if name == "stats":
        warnings.warn(
            "neworder.stats is deprecated and will be removed in a future release. "
            "Use scipy.special.expit and scipy.special.logit instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import _neworder_core  # ty:ignore[unresolved-import]

        return _neworder_core.stats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
