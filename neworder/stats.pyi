"""

Submodule for statistical functions
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = ["logistic", "logit"]

@typing.overload
def logistic(
    x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], x0: typing.SupportsFloat, k: typing.SupportsFloat
) -> numpy.typing.NDArray[numpy.float64]:
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
def logistic(
    x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], k: typing.SupportsFloat
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Computes the logistic function with x0=0 on the supplied values.
    Args:
        x: The input values.
        k: The growth rate
    Returns:
        The function values
    """

@typing.overload
def logistic(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Computes the logistic function with k=1 and x0=0 on the supplied values.
    Args:
        x: The input values.
    Returns:
        The function values
    """

def logit(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Computes the logit function on the supplied values.
    Args:
        x: The input probability values in (0,1).
    Returns:
        The function values (log-odds)
    """
