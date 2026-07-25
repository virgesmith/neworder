from math import sqrt

import numpy as np
import pandas as pd
import pytest

import neworder as no


def test_errors() -> None:
    df = pd.read_csv("./test/df.csv")
    df["DC2101EW_C_ETHPUK11"] = pd.Categorical(df["DC2101EW_C_ETHPUK11"], categories=range(4))

    # base model for MC engine
    model = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

    m = len(df["DC2101EW_C_ETHPUK11"].cat.categories)
    # identity matrix means no transitions
    trans = np.identity(m)

    # invalid transition matrices
    with pytest.raises(ValueError):
        no.df.transition(model, np.ones((1, 2)), df, "DC2101EW_C_ETHPUK11")
    with pytest.raises(ValueError):
        no.df.transition(model, np.ones((1, 1)), df, "DC2101EW_C_ETHPUK11")
    with pytest.raises(ValueError):
        no.df.transition(model, trans + 0.1, df, "DC2101EW_C_ETHPUK11")

    # a plain int64 column (not a pandas "category" dtype) is not supported
    df["intcol"] = np.zeros(len(df), dtype=np.int64)
    with pytest.raises(TypeError):
        no.df.transition(model, trans, df, "intcol")

    # nor is a plain (non-categorical) object/string column
    df["strcol"] = "x"
    with pytest.raises(TypeError):
        no.df.transition(model, trans, df, "strcol")


def test_categorical(base_model: no.Model) -> None:
    N = 100000

    # string category labels via pandas "category" dtype
    df = pd.DataFrame({"region": pd.Categorical(["north"] * N, categories=["north", "south", "east", "west"])})

    # deterministic north -> south
    t = np.identity(4)
    t[0, 0] = 0.0
    t[0, 1] = 1.0
    df["region"] = no.df.transition(base_model, t, df, "region")
    assert df["region"].dtype == "category"
    assert list(df["region"].cat.categories) == ["north", "south", "east", "west"]
    assert df.region.value_counts()["south"] == N
    assert df.region.value_counts()["north"] == 0

    # spread evenly among all 4 categories
    t2 = np.ones((4, 4)) / 4
    df["region"] = no.df.transition(base_model, t2, df, "region")
    for cat in ["north", "south", "east", "west"]:
        assert df.region.value_counts()[cat] > N / 4 - sqrt(N) and df.region.value_counts()[cat] < N / 4 + sqrt(N)

    # transition matrix size must match the number of pandas categories
    with pytest.raises(ValueError):
        no.df.transition(base_model, np.identity(2), df, "region")

    # NaN/missing categories (code -1) are left untouched
    df_nan = pd.DataFrame({"region": pd.Categorical(["north", None, "south", None], categories=["north", "south"])})
    df_nan["region"] = no.df.transition(base_model, np.identity(2), df_nan, "region")
    assert df_nan["region"].cat.codes.tolist() == [0, -1, 1, -1]

    # the ordered flag is preserved
    df_ord = pd.DataFrame({"grade": pd.Categorical(["A", "B"], categories=["A", "B", "C"], ordered=True)})
    df_ord["grade"] = no.df.transition(base_model, np.identity(3), df_ord, "grade")
    assert df_ord["grade"].cat.ordered is True
    assert list(df_ord["grade"].cat.categories) == ["A", "B", "C"]


def test_basic() -> None:
    # test unique index generation
    idx = no.df.unique_index(100)
    assert np.array_equal(idx, np.arange(no.mpi.RANK, 100 * no.mpi.SIZE, step=no.mpi.SIZE))

    idx = no.df.unique_index(100)
    assert np.array_equal(
        idx,
        np.arange(100 * no.mpi.SIZE + no.mpi.RANK, 200 * no.mpi.SIZE, step=no.mpi.SIZE),
    )

    N = 100000
    # base model for MC engine
    model = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

    df = pd.DataFrame({"category": pd.Categorical([1] * N, categories=[1, 2, 3])})

    # no transitions, check no changes
    t = np.identity(3)
    df["category"] = no.df.transition(model, t, df, "category")
    assert df.category.value_counts()[1] == N

    # all 1 -> 2
    t[0, 0] = 0.0
    t[0, 1] = 1.0
    df["category"] = no.df.transition(model, t, df, "category")
    assert df.category.value_counts()[1] == 0
    assert df.category.value_counts()[2] == N

    # 2 -> 1 or 3
    t = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )

    df["category"] = no.df.transition(model, t, df, "category")
    assert df.category.value_counts()[2] == 0
    for i in [1, 3]:
        assert df.category.value_counts()[i] > N / 2 - sqrt(N) and df.category.value_counts()[i] < N / 2 + sqrt(N)

    # spread evenly
    t = np.ones((3, 3)) / 3
    df["category"] = no.df.transition(model, t, df, "category")
    for i in [1, 2, 3]:
        assert df.category.value_counts()[i] > N / 3 - sqrt(N) and df.category.value_counts()[i] < N / 3 + sqrt(N)

    # all -> 1
    t = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    df["category"] = no.df.transition(model, t, df, "category")
    assert df.category.value_counts()[1] == N


def test(base_model: no.Model) -> None:
    df = pd.read_csv("./test/df.csv")
    df["DC2101EW_C_ETHPUK11"] = pd.Categorical(df["DC2101EW_C_ETHPUK11"], categories=range(4))

    # identity matrix means no transitions
    trans = np.identity(4)

    df["DC2101EW_C_ETHPUK11"] = no.df.transition(base_model, trans, df, "DC2101EW_C_ETHPUK11")

    assert len(df["DC2101EW_C_ETHPUK11"].unique()) == 1 and df["DC2101EW_C_ETHPUK11"].unique()[0] == 2

    # NOTE transition matrix interpreted as being COLUMN MAJOR due to pandas DataFrame storing data in column-major order

    # force 2->3
    trans[2, 2] = 0.0
    trans[2, 3] = 1.0
    df["DC2101EW_C_ETHPUK11"] = no.df.transition(base_model, trans, df, "DC2101EW_C_ETHPUK11")
    no.log(df["DC2101EW_C_ETHPUK11"].unique())
    assert len(df["DC2101EW_C_ETHPUK11"].unique()) == 1 and df["DC2101EW_C_ETHPUK11"].unique()[0] == 3

    # ~half of 3->0
    trans[3, 0] = 0.5
    trans[3, 3] = 0.5
    df["DC2101EW_C_ETHPUK11"] = no.df.transition(base_model, trans, df, "DC2101EW_C_ETHPUK11")
    assert np.array_equal(np.sort(df["DC2101EW_C_ETHPUK11"].unique()), np.array([0, 3]))
