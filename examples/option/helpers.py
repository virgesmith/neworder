from typing import TypeVar

import numpy as np
import scipy.special  # type: ignore

# for inverse cumulative normal
import scipy.stats  # type: ignore

T = TypeVar("T")  # Any type.
nparray = np.ndarray[T, np.dtype[T]]


def nstream(u: nparray[np.float64]) -> nparray[np.float64]:
    """Return a vector of n normally distributed pseudorandom variates (mean zero unit variance)"""
    return scipy.stats.norm.ppf(u)


def norm_cdf(x: nparray[np.float64]) -> nparray[np.float64]:
    """Compute the normal cumulatve density funtion"""
    return (1.0 + scipy.special.erf(x / np.sqrt(2.0))) / 2.0
