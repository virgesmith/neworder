import importlib.metadata

__version__ = importlib.metadata.version("neworder")

from _neworder_core import (
    CalendarTimeline,
    LinearTimeline,
    Model,
    MonteCarlo,
    NoTimeline,
    NumericTimeline,
    Timeline,
    checked,
    df,
    log,
    mpi,
    run,
    stats,
    time,
    verbose,
)

# type: ignore
from .domain import Domain, Edge, Space, StateGrid
from .mc import as_np

__all__ = ["as_np", "Domain", "Space"]
