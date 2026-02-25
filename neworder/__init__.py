import importlib.metadata

__version__ = importlib.metadata.version("neworder")

from _neworder_core import (
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
