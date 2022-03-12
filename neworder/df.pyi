"""
    Submodule for operations involving direct manipulation of pandas dataframes
"""
from __future__ import annotations
from typing import TypeVar
import neworder
import numpy as np
# _Shape = typing.Tuple[int, ...]

T = TypeVar("T")
nparray = np.ndarray[T, np.dtype[T]]

__all__ = [
    "testfunc",
    "transition",
    "unique_index"
]


def testfunc(model: neworder.Model, df: object, colname: str) -> None:
    """
    Test function for direct dataframe manipulation. Results may vary. Do not use.
    """
def transition(model: neworder.Model, categories: nparray[np.int64], transition_matrix: nparray[np.float64], df: object, colname: str) -> None:
    """
    Randomly changes categorical data in a dataframe, according to supplied transition probabilities.
    Args:
        model: The model (for access to the MonteCarlo engine).
        categories: The set of possible categories
        transition_matrix: The probabilities of transitions between categories
        df: The dataframe, which is modified in-place
        colname: The name of the column in the dataframe
    """
def unique_index(n: int) -> nparray[np.int64]:
    """
    Generates an array of n unique values, even across multiple processes, that can be used to unambiguously index multiple dataframes.
    """
