""" helpers.py for RiskPaths """

import numpy as np
from bisect import bisect_left

def partition(start, finish, step=1):
  """ Helper function to return an inclusive equal-spaced range, i.e. finish will be the last element """
  # ensure finish is always included
  return np.append(np .arange(start, finish, step), finish)


def interp(rng, value):
  """ Equivalent to self-scheduling split """
  # TODO check behaviour outside range is same
  idx = bisect_left(rng, value)
  # if idx == len(range)
  #   idx = idx - 1
  return idx