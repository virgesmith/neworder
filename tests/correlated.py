""" MPI tests """

import numpy as np
import pandas as pd
import neworder

import test as test_


def test():
  t = test_.Test()

  if neworder.mpi.size() == 1:
    neworder.log("Skipping MPI tests")
    return True

  # base model for testing MC engine
  model = neworder.Model(neworder.Timeline.null())
  neworder.log(model)
  
  # test ustream/sequence
  t.check(not neworder.mpi.indep())

  u = model.mc().ustream(1000)
  v = neworder.mpi.broadcast(u, 0)
  # u == v for all processes

  t.check(np.array_equal(u, v))

  return not t.any_failed