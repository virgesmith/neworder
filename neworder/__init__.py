import importlib.metadata

__version__ = importlib.metadata.version("neworder")

from _neworder_core import (  # ty:ignore[import-not-found]
    Model,
    MonteCarlo,
    SplitMix64,
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

from . import stats, timeline
from .domain import Domain, Edge, Space, StateGrid
from .mc import as_np
from .timeline import Timestep

__all__: list[str] = [
    "Domain",
    "Edge",
    "Model",
    "MonteCarlo",
    "Space",
    "SplitMix64",
    "StateGrid",
    "Timestep",
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
    "timeline",
    "verbose",
]
