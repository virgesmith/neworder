"""

Submodule for statistical functions
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing as npt

__all__ = ["logistic", "logit"]

@typing.overload
def logistic(x: npt.NDArray[numpy.float64], x0: float, k: float) -> npt.NDArray[numpy.float64]:
    """
    Computes the logistic function on the supplied values.
    Args:
        x: The input values.
        k: The growth rate
        x0: the midpoint location
    Returns:
        The function values
    """

@typing.overload
def logistic(x: npt.NDArray[numpy.float64], k: float) -> npt.NDArray[numpy.float64]:
    """
    Computes the logistic function with x0=0 on the supplied values.
    Args:
        x: The input values.
        k: The growth rate
    Returns:
        The function values
    """

@typing.overload
def logistic(x: npt.NDArray[numpy.float64]) -> npt.NDArray[numpy.float64]:
    """
    Computes the logistic function with k=1 and x0=0 on the supplied values.
    Args:
        x: The input values.
    Returns:
        The function values
    """

def logit(x: npt.NDArray[numpy.float64]) -> npt.NDArray[numpy.float64]:
    """
    Computes the logit function on the supplied values.
    Args:
        x: The input probability values in (0,1).
    Returns:
        The function values (log-odds)
    """
