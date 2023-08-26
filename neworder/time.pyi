"""
    Temporal values and comparison.
"""
from __future__ import annotations
import typing
import numpy as np
import numpy.typing as npt

__all__ = [
    "DISTANT_PAST",
    "FAR_FUTURE",
    "NEVER",
    "isnever"
]


@typing.overload
def isnever(t: float) -> bool:
    """
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN,
    direct comparison will always fail, since NaN != NaN.
    """

@typing.overload
def isnever(t: npt.NDArray[np.float64] | list[float]) -> npt.NDArray[np.bool8]:
    """
    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN.
    """
DISTANT_PAST: float # value = -inf
"""A value that compares less than any other value but itself and NEVER"""

FAR_FUTURE: float # value = inf
"""A value that compares greater than any other value but itself and NEVER"""

NEVER: float # value = nan
"""A value that compares unequal to any value, including itself"""
