import pytest
import numpy as np
import neworder as no

#no.verbose()


def test_basics():
  # just check you can call the functions
  no.version()
  no.python()
  no.log("testing")
  assert not no.embedded()

def test_mpi():
  # if no mpi4py, assume serial like module does
  try:
    import mpi4py.MPI as mpi
    rank = mpi.COMM_WORLD.Get_rank()
    size = mpi.COMM_WORLD.Get_size()
  except Exception:
    rank = 0
    size = 1
  assert no.mpi.rank() == rank
  assert no.mpi.size() == size


