
from math import erf, sqrt, log, exp
import numpy as np
# for inverse cumulative normal
import scipy.stats

import neworder

from humanleague import sobolSequence 

def nstream(n):
  """ Return a vector of n normally distributed pseudorandom variates (mean zero unity variance) """
  return scipy.stats.norm.ppf(neworder.ustream(n))

def nstream_q(n):
  """ Return a vector of n normally distributed quasirandom variates (mean zero unity variance) """
  return scipy.stats.norm.ppf(sobolSequence(1, n, n * neworder.seq))

def norm_cdf(x):
  """ Compute the inverse normal cumulatve density funtion """
  return (1.0 + erf(x / sqrt(2.0))) / 2.0

