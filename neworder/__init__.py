import importlib.metadata

__version__ = importlib.metadata.version("neworder")

from _neworder_core import (  # ty:ignore[unresolved-import]
    Model,
    MonteCarlo,
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

from . import stats
from .domain import Domain, Edge, Space, StateGrid
from .mc import as_np
from .timeline import CalendarTimeline, LinearTimeline, NoTimeline, NumericTimeline

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
