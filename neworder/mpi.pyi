"""
    Submodule for basic MPI environment discovery.
"""
from __future__ import annotations
from typing import Any

__all__ = [
    "COMM",
    "RANK",
    "SIZE"
]

COMM: Any
"""The MPI communicator if neworder has been installed with the parallel option, otherwise None."""

RANK: int
"""The MPI process rank. 0 in serial mode."""

SIZE: int
"""The number of MPI processes. 1 in serial mode"""
