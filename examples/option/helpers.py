
from math import erf, sqrt, log, exp
import numpy as np
# for inverse cumulative normal
import scipy.stats

import neworder

# TODO move this functionality into C++

def nstream(u):
  """ Return a vector of n normally distributed pseudorandom variates (mean zero unity variance) """
  return scipy.stats.norm.ppf(u)

def norm_cdf(x):
  """ Compute the normal cumulatve density funtion """
  return (1.0 + erf(x / sqrt(2.0))) / 2.0

