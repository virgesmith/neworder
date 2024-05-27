import time

import numpy as np
import pandas as pd  # type: ignore

import neworder as no

no.verbose()

# define some global variables describing where the starting population and the parameters of the dynamics come from
INITIAL_POPULATION = "./ssm_hh_E09000001_OA11_2011.csv"

t = np.array(
    [
        [0.9, 0.05, 0.05, 0.0, 0.0, 0.0],
        [0.05, 0.9, 0.04, 0.01, 0.0, 0.0],
        [0.0, 0.05, 0.9, 0.05, 0.0, 0.0],
        [0.0, 0.0, 0.05, 0.9, 0.05, 0.0],
        [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.8],
    ]
)

c = np.array([-1, 1, 2, 3, 4, 5])


def get_data() -> pd.DataFrame:
    hh = pd.read_csv(INITIAL_POPULATION)  # , nrows=100)
    hh = pd.concat([hh] * 8, ignore_index=True)
    return hh


def interp(cumprob: np.ndarray[np.float64, np.dtype[np.float64]], x: float) -> int:
    lbound = 0
    while lbound < len(cumprob) - 1:
        if cumprob[lbound] > x:
            break
        lbound += 1
    return lbound


def sample(
    u: float,
    tc: np.ndarray[np.float64, np.dtype[np.float64]],
    c: np.ndarray[np.float64, np.dtype[np.float64]],
) -> float:
    return c[interp(tc, u)]


def transition(
    c: np.ndarray[np.float64, np.dtype[np.float64]],
    t: np.ndarray[np.float64, np.dtype[np.float64]],
    df: pd.DataFrame,
    colname: str,
) -> None:
    # u = m.mc.ustream(len(df))
    tc = np.cumsum(t, axis=1)

    # reverse mapping of category label to index
    lookup = {c[i]: i for i in range(len(c))}

    # for i in range(len(df)):
    #   current = df.loc[i, colname]
    #   df.loc[i, colname] = sample(u[i], tc[lookup[current]], c)

    df[colname] = df[colname].apply(
        lambda current: sample(m.mc.ustream(1)[0], tc[lookup[current]], c)
    )


def python_impl(m: no.Model, df: pd.DataFrame) -> tuple[int, float, pd.Series]:
    start = time.time()
    transition(c, t, df, "LC4408_C_AHTHUK11")
    return len(df), time.time() - start, df.LC4408_C_AHTHUK11


def cpp_impl(m: no.Model, df: pd.DataFrame) -> tuple[int, float, pd.Series]:
    start = time.time()
    no.df.transition(m, c, t, df, "LC4408_C_AHTHUK11")
    return len(df), time.time() - start, df.LC4408_C_AHTHUK11


# def f(m):

# n = 1000

# c = [1,2,3]
# df = pd.DataFrame({"n": [1]*n})

# # no transitions
# t = np.identity(3)

# no.df.transition(m, c, t, df, "n")
# no.log(df.n.value_counts()[1] == 1000)

# # all 1 -> 2
# t[0,0] = 0.0
# t[1,0] = 1.0
# no.df.transition(m, c, t, df, "n")
# no.log(df.n.value_counts()[2] == 1000)

# # all 2 -> 1 or 3
# t = np.array([
#   [1.0, 0.5, 0.0],
#   [0.0, 0.0, 0.0],
#   [0.0, 0.5, 1.0],
# ])

# no.df.transition(m, c, t, df, "n")
# no.log(2 not in df.n.value_counts())#[2] == 1000)
# no.log(df.n.value_counts())

# t = np.ones((3,3)) / 3
# no.df.transition(m, c, t, df, "n")
# no.log(df.n.value_counts())
# for i in c:
#   no.log(df.n.value_counts()[i] > n/3 - sqrt(n) and df.n.value_counts()[i] < n/3 + sqrt(n))

# t = np.array([
#   [1.0, 1.0, 1.0],
#   [0.0, 0.0, 0.0],
#   [0.0, 0.0, 0.0],
# ])
# no.df.transition(m, c, t, df, "n")
# no.log(df.n.value_counts())

if __name__ == "__main__":
    m = no.Model(no.NoTimeline(), no.MonteCarlo.deterministic_identical_stream)

    rows, tc, colcpp = cpp_impl(m, get_data())
    no.log("C++ %d: %f" % (rows, tc))

    m.mc.reset()
    rows, tp, colpy = python_impl(m, get_data())
    no.log("py  %d: %f" % (rows, tp))

    # no.log(colcpp-colpy)

    assert np.array_equal(colcpp, colpy)

    no.log("speedup factor = %f" % (tp / tc))

#  f(m)
