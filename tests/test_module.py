import pytest
import numpy as np
import neworder as no

#no.verbose()


def test_basics():
  with open("VERSION") as f:
    assert f.readline().rstrip() == no.version()

  # just check you can call the functions
  no.python()
  assert hasattr(no, "verbose")
  no.log("testing")
  no.log(1)
  no.log(no)
  no.log([1,2,3])
  no.log((1,2,3))
  no.log({1:2,3:4})

def test_submodules():
  assert(hasattr(no, "mpi"))
  assert(hasattr(no, "stats"))
  assert(hasattr(no, "dataframe"))

def test_dummy_model():
  class DummyModel(no.Model):
    def __init__(self):
      super().__init__(no.Timeline.null(), no.MonteCarlo.deterministic_identical_seed)
    def step(self):
      pass
    def checkpoint(self):
      pass
  assert no.run(DummyModel())

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


