"""
Submodule for statistical functions

!!! warning "Deprecated"
    This module will be removed in a future release. Use the `scipy.special` equivalents instead.
"""

from typing import SupportsFloat
from warnings import deprecated

import numpy as np
import numpy.typing as npt
from _neworder_core import stats as _stats  # ty:ignore[unresolved-import]

__all__: list[str] = ["logistic", "logit"]


@deprecated("neworder.stats.logistic is deprecated, use scipy.special.expit(k * (x - x0)) instead")
def logistic(x: npt.ArrayLike, x0: SupportsFloat = 0.0, k: SupportsFloat = 1.0) -> npt.NDArray[np.float64]:
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
    return _stats.logistic(x, x0, k)


@deprecated("neworder.stats.logit is deprecated, use scipy.special.logit instead")
def logit(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Computes the logit function on the supplied values.

    !!! warning "Deprecated"
        Use `scipy.special.logit(x)` instead.

    Args:
        x: The input probability values in (0,1).
    Returns:
        The function values (log-odds)
    """
    return _stats.logit(x)
