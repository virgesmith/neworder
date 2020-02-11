""" MPI tests """

import numpy as np
import pandas as pd
import neworder

import test as test_


def test():
  t = test_.Test()

  if neworder.size() == 1:
    neworder.log("Skipping MPI tests")
    return True
  
  # test ustream/sequence
  t.check(not neworder.mc.indep())

  u = neworder.mc.ustream(1000)
  v = neworder.broadcast(u, 0)
  # u == v for all processes

  t.check(np.array_equal(u, v))

  return not t.any_failed