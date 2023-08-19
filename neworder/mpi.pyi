"""
    Submodule for basic MPI environment discovery, containing the following attributes:

    rank: the process rank (0 in serial mode)
    size: the number of processes (1 in serial mode)
    comm: the MPI communicator (None in serial mode)
"""
from __future__ import annotations
try:
    import mpi4py.MPI
    MPI_comm_t = mpi4py.MPI.Intracomm
except ImportError:
    MPI_comm_t = type(None)


__all__ = [
    "comm",
    "rank",
    "size"
]

comm: MPI_comm_t
"""The MPI communicator, or None """

rank: int
"""The MPI process rank ()"""

size: int
"""The number of MPI processes"""
