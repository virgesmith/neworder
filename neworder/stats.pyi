"""
    Submodule for statistical functions
"""
from __future__ import annotations
from typing import overload, TypeVar
import numpy as np
#_Shape = typing.Tuple[int, ...]

__all__ = [
    "logistic",
    "logit"
]

T = TypeVar("T")
nparray = np.ndarray[T, np.dtype[T]]

@overload
def logistic(x: nparray[np.float64]) -> nparray[np.float64]:
    """
    Computes the logistic function on the supplied values.
    Args:
        x: The input values.
        k: The growth rate
        x0: the midpoint location
    Returns:
        The function values


    Computes the logistic function with x0=0 on the supplied values.
    Args:
        x: The input values.
        k: The growth rate
    Returns:
        The function values


    Computes the logistic function with k=1 and x0=0 on the supplied values.
    Args:
        x: The input values.
    Returns:
        The function values
    """
@overload
def logistic(x: nparray[np.float64], k: float) -> nparray[np.float64]:
    pass
@overload
def logistic(x: nparray[np.float64], x0: float, k: float) -> nparray[np.float64]:
    pass
def logit(x: nparray[np.float64]) -> nparray[np.float64]:
    """
    Computes the logit function on the supplied values.
    Args:
        x: The input probability values in (0,1).
    Returns:
        The function values (log-odds)
    """
