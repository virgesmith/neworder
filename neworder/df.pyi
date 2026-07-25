"""

Submodule for operations involving direct manipulation of pandas dataframes
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

import neworder

__all__: list[str] = ["testfunc", "transition", "unique_index"]

def testfunc(model: neworder.Model, df: typing.Any, colname: str) -> None:
    """
    Test function for direct dataframe manipulation. Results may vary. Do not use.
    """

def transition(
    model: neworder.Model,
    transition_matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    df: typing.Any,
    colname: str,
) -> None:
    """
    Randomly changes categorical data in a dataframe, according to supplied transition probabilities.
    The column must have a pandas "category" dtype (any category label type, e.g. strings, is supported) -
    convert it first with df[colname] = df[colname].astype("category") if necessary. The row order of
    transition_matrix must correspond to the column's cat.categories order.
    Args:
        model: The model (for access to the MonteCarlo engine).
        transition_matrix: The probabilities of transitions between categories
        df: The dataframe, which is modified in-place
        colname: The name of the column in the dataframe
    """

def unique_index(n: typing.SupportsInt, offset: typing.SupportsInt = 0) -> numpy.typing.NDArray[numpy.int64]:
    """
    Generates an array of n unique values, even across multiple processes, that can be used to unambiguously index multiple dataframes.
    When multiple threads are in use, a unique offset must be specifically provided for each thread. (Using the thread
    id is generally nondeterministic)
    """
