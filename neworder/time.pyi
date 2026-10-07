"""

Temporal values and comparison, including the attributes:
NEVER: a value that compares unequal to any value, including itself.
DISTANT_PAST: a value that compares less than any other value but itself and NEVER
FAR_FUTURE: a value that compares greater than any other value but itself and NEVER
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = ["DISTANT_PAST", "FAR_FUTURE", "NEVER", "isnever"]

@typing.overload
def isnever(t: typing.SupportsFloat) -> bool:
    """
    Returns whether the value of t corresponds to "never". As "never" is implemented as a floating-point NaN,
    direct comparison will always fail, since NaN != NaN.
    """

@typing.overload
def isnever(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.bool]:
    """
    Returns an array of booleans corresponding to whether the element of an array correspond to "never". As "never" is
    implemented as a floating-point NaN, direct comparison will always fails, since NaN != NaN.
    """

# Not generated: griffe drops overloads that have no implementation signature, which hides them from the API docs
def isnever(
    t: typing.SupportsFloat | typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> bool | numpy.typing.NDArray[numpy.bool]:
    """
    Returns whether the value of t (scalar or array) corresponds to "never". As "never" is implemented as a
    floating-point NaN, direct comparison will always fail, since NaN != NaN.
    Args:
        t: The time value(s) to test.
    Returns:
        A bool for scalar input, or an array of bools for array input.
    """

DISTANT_PAST: float  # value = -inf
"""A value that compares less than any other value but itself and NEVER"""
FAR_FUTURE: float  # value = inf
"""A value that compares greater than any other value but itself and NEVER"""
NEVER: float  # value = nan
"""A value that compares unequal to any value, including itself. Use isnever() to test for it"""
