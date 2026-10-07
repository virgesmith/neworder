"""

Submodule for statistical functions

!!! warning "Deprecated"
    `neworder.stats` is deprecated and will be removed in a future release. Use the equivalent functions in
    `scipy.special` instead:

    - `logistic(x, x0, k)` is `scipy.special.expit(k * (x - x0))`
    - `logit(x)` is `scipy.special.logit(x)`
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing
from typing_extensions import deprecated

__all__: list[str] = ["logistic", "logit"]

@deprecated("neworder.stats is deprecated, use scipy.special.expit(k * (x - x0)) instead")
def logistic(
    x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x0: typing.SupportsFloat = 0.0,
    k: typing.SupportsFloat = 1.0,
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Computes the logistic function on the supplied values.
    Args:
        x: The input values.
        x0: the midpoint location (default 0)
        k: The growth rate (1/scale, default 1)
    Returns:
        The function values
    """

@deprecated("neworder.stats is deprecated, use scipy.special.logit instead")
def logit(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Computes the logit function on the supplied values.
    Args:
        x: The input probability values in (0,1).
    Returns:
        The function values (log-odds)
    """
