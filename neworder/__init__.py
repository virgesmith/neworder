import importlib.metadata

__version__ = importlib.metadata.version("neworder")

from _neworder_core import (  # ty:ignore[unresolved-import]
    LinearTimeline,
    Model,
    MonteCarlo,
    NoTimeline,
    NumericTimeline,
    Timeline,
    checked,
    df,
    freethreaded,
    log,
    mpi,
    run,
    stats,
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
