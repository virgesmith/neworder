"""
    Submodule for basic MPI environment discovery
"""
from __future__ import annotations

__all__ = [
    "rank",
    "size"
]


def rank() -> int:
    """
    Returns the MPI rank of the process
    """
def size() -> int:
    """
    Returns the MPI size (no. of processes) of the run
    """
