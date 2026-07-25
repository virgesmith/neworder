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
) -> typing.Any:
    """
    Randomly changes categorical data, according to supplied transition probabilities, and returns the result
    as a new pandas Categorical - it does not modify df in-place, so the caller is responsible for assigning
    the result back, e.g. df[colname] = no.df.transition(model, transition_matrix, df, colname).
    The column must have a pandas "category" dtype (any category label type, e.g. strings, is supported) -
    convert it first with df[colname] = df[colname].astype("category") if necessary. The row order of
    transition_matrix must correspond to the column's cat.categories order.
    Args:
        model: The model (for access to the MonteCarlo engine).
        transition_matrix: The probabilities of transitions between categories
        df: The dataframe containing the column to transition
        colname: The name of the column in the dataframe
    Returns:
        The transitioned data, as a new pandas Categorical with the same categories/order as df[colname].
    """

def unique_index(n: typing.SupportsInt, offset: typing.SupportsInt = 0) -> numpy.typing.NDArray[numpy.int64]:
    """
    Generates an array of n unique values, even across multiple processes, that can be used to unambiguously index multiple dataframes.
    When multiple threads are in use, a unique offset must be specifically provided for each thread. (Using the thread
    id is generally nondeterministic)
    """
