"""

Submodule for operations involving direct manipulation of pandas dataframes
"""

from __future__ import annotations

import typing
from collections.abc import Mapping

import numpy
import numpy.typing
import pandas

import neworder

__all__: list[str] = ["transition", "transition_conditional", "unique_index"]

def transition(
    mc: neworder.MonteCarlo,
    transition_matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    series: pandas.Series,
) -> pandas.Categorical:
    """
    Randomly changes categorical data, according to supplied transition probabilities, and returns the result
    as a new pandas Categorical - it does not modify series in-place, so the caller is responsible for
    assigning the result back, e.g. df[colname] = no.df.transition(model.mc, transition_matrix, df[colname]).
    The series must have a pandas "category" dtype (any category label type, e.g. strings, is supported) -
    convert it first with series = series.astype("category") if necessary. The row order of transition_matrix
    must correspond to the series' cat.categories order.
    Args:
        mc: The model's MonteCarlo engine (e.g. model.mc).
        transition_matrix: The probabilities of transitions between categories
        series: The pandas Series (categorical dtype) to transition
    Returns:
        The transitioned data, as a new pandas Categorical with the same categories/order as series.
    """

def transition_conditional(
    mc: neworder.MonteCarlo,
    matrices: Mapping[typing.Any, typing.Annotated[numpy.typing.ArrayLike, numpy.float64]],
    group: pandas.Series,
    series: pandas.Series,
) -> pandas.Categorical:
    """
    Like transition(), but applies a different transition matrix per row depending on the corresponding value
    of another categorical column (group), e.g. transition probabilities that vary by age band or sex. It does
    not modify series in-place, so the caller is responsible for assigning the result back, e.g.
    df[colname] = no.df.transition_conditional(model.mc, matrices, df[groupname], df[colname]).
    Both series and group must have a pandas "category" dtype, and must be the same length and row-aligned -
    convert them first with series = series.astype("category") if necessary. matrices is a dict mapping each of
    group's category labels to the (square) transition matrix to apply to rows in that group; its row order
    must correspond to series' cat.categories order, as in transition(). Rows whose group value is NaN/missing
    are left untouched; every other category present in group.cat.categories must have a corresponding entry
    in matrices.
    Args:
        mc: The model's MonteCarlo engine (e.g. model.mc).
        matrices: dict mapping each category in group to the transition matrix (probabilities of transitions
            between categories) to apply where group has that value
        group: The pandas Series (categorical dtype) whose value selects the transition matrix per row
        series: The pandas Series (categorical dtype) to transition
    Returns:
        The transitioned data, as a new pandas Categorical with the same categories/order as series.
    """

def unique_index(n: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
    """
    Generates an array of n unique values, even across multiple processes, that can be used to unambiguously index multiple dataframes.
    When multiple threads are in use, specific index values should not be relied on as they are generally nondeterministic
    """
