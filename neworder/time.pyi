"""
    Temporal values and comparison
"""
from __future__ import annotations
import typing
import numpy as np
import numpy.typing as npt

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
def never() -> float:
    """
    Returns a value that compares unequal to any value, including itself.
    """
