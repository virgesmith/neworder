"""

Submodule for statistical functions
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing
from typing_extensions import deprecated

__all__: list[str] = ["logistic", "logit"]

@deprecated("neworder.stats.logistic is deprecated, use scipy.special.expit(k * (x - x0)) instead")
def logistic(
    x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x0: typing.SupportsFloat = 0.0,
    k: typing.SupportsFloat = 1.0,
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Computes the logistic function on the supplied values.

    !!! warning "Deprecated"
        Use `scipy.special.expit(k * (x - x0))` instead.

    Args:
        x: The input values.
        x0: the midpoint location (default 0)
        k: The growth rate (1/scale, default 1)
    Returns:
        The function values
    """

@deprecated("neworder.stats.logit is deprecated, use scipy.special.logit instead")
def logit(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Computes the logit function on the supplied values.

    !!! warning "Deprecated"
        Use `scipy.special.logit(x)` instead.

    Args:
        x: The input probability values in (0,1).
    Returns:
        The function values (log-odds)
    """
