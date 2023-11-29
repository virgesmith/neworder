import importlib.metadata

__version__ = importlib.metadata.version("neworder")

from _neworder_core import (
    Model,
    MonteCarlo,
    Timeline,
    NoTimeline,
    LinearTimeline,
    NumericTimeline,
    CalendarTimeline,
    time,
    df,
    mpi,
    log,
    run,
    stats,
    checked,
    verbose,
)  # type: ignore
from .domain import Edge, Domain, Space, StateGrid
from .mc import as_np
