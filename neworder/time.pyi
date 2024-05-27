"""

Temporal values and comparison, including the attributes:
NEVER: a value that compares unequal to any value, including itself.
DISTANT_PAST: a value that compares less than any other value but itself and NEVER
FAR_FUTURE: a value that compares greater than any other value but itself and NEVER
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing as npt

__all__ = ["DISTANT_PAST", "FAR_FUTURE", "NEVER", "isnever"]

@typing.overload
def isnever(t: float) -> bool:
    """
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN,
    direct comparison will always fail, since NaN != NaN.
    """

@typing.overload
def isnever(t: npt.NDArray[numpy.float64]) -> npt.NDArray:
    """
    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN.
    """

DISTANT_PAST: float  # value = -inf
FAR_FUTURE: float  # value = inf
NEVER: float  # value = nan
