"""

Submodule for basic MPI environment discovery, containing the following attributes:

RANK: the process rank (0 in serial mode)
SIZE: the number of processes (1 in serial mode)
COMM: the MPI communicator (None in serial mode)
"""

from __future__ import annotations

import mpi4py.MPI  # type: ignore[import-not-found]

__all__ = ["COMM", "RANK", "SIZE"]
COMM: mpi4py.MPI.Intracomm  # value = <mpi4py.MPI.Intracomm object>
RANK: int = 0
SIZE: int = 1
