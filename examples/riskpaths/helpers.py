""" helpers.py for RiskPaths """

import numpy as np
from bisect import bisect_left

def partition(start, finish, step=1):
  """ Helper function to return an inclusive equal-spaced range, i.e. finish will be the last element """
  return np.linspace(start, finish, (finish-start)/step + 1)

def interp(range, value):
  """ Equivalent to self-scheduling split """
  # TODO check behaviour outside range is same
  idx = bisect_left(range, value)
  # if idx == len(range)
  #   idx = idx - 1
  return idx