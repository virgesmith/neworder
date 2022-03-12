"""
    Temporal values and comparison
"""
from __future__ import annotations
from typing import overload, TypeVar
import numpy as np

T = TypeVar("T")
nparray = np.ndarray[T, np.dtype[T]]


__all__ = [
    "distant_past",
    "far_future",
    "isnever",
    "never"
]


def distant_past() -> float:
    """
    Returns a value that compares less than any other value but itself and "never"
    """
def far_future() -> float:
    """
    Returns a value that compares greater than any other value but itself and "never"
    """
@overload
def isnever(t: float) -> bool:
    """
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN,
    direct comparison will always fail, since NaN != NaN.


    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN.
    """
@overload
def isnever(t: nparray[np.float64]) -> nparray[np.bool8]:
    pass
def never() -> float:
    """
    Returns a value that compares unequal to any value, including but itself.
    """
