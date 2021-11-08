
import numpy as np
# for inverse cumulative normal
import scipy.stats
import scipy.special


def nstream(u):
  """ Return a vector of n normally distributed pseudorandom variates (mean zero unit variance) """
  return scipy.stats.norm.ppf(u)


def norm_cdf(x):
  """ Compute the normal cumulatve density funtion """
  return (1.0 + scipy.special.erf(x / np.sqrt(2.0))) / 2.0

