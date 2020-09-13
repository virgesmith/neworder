
import numpy as np
import pandas as pd
import neworder as no
from math import sqrt

# def sample(u, t, c):
#   i = int(np.interp(u, t, range(len(t))))
#   return c[i]

def test_errors():

  df = pd.read_csv("./tests/df.csv")

  # base model for MC engine
  model = no.Model(no.Timeline.null(), no.MonteCarlo.deterministic_identical_stream)

  cats = np.array(range(4)) 
  # identity matrix means no transitions
  trans = np.identity(len(cats))

  # category data MUST be 64bit integer. This will alomst certainly be the default on linux/OSX but not on windows
  df["DC2101EW_C_ETHPUK11"]= df["DC2101EW_C_ETHPUK11"].astype(np.int32)

  try:
    no.df.transition(model, cats, trans, df, "DC2101EW_C_ETHPUK11")
  except Exception:
    assert True
  else:
    assert False

def test_basic():

  # test unique index generation
  idx = no.df.unique_index(100)
  assert np.array_equal(idx, np.arange(no.mpi.rank(), 100 * no.mpi.size(), step=no.mpi.size()))

  N = 100000
  # base model for MC engine
  model = no.Model(no.Timeline.null(), no.MonteCarlo.deterministic_identical_stream)

  c = [1,2,3]
  df = pd.DataFrame({"category": [1] * N})

  # no transitions, check no changes
  t = np.identity(3)
  no.df.transition(model, c, t, df, "category")
  assert df.category.value_counts()[1] == N

  # all 1 -> 2
  t[0,0] = 0.0
  t[0,1] = 1.0
  no.df.transition(model, c, t, df, "category")
  assert 1 not in df.category.value_counts()
  assert df.category.value_counts()[2] == N

  # 2 -> 1 or 3
  t = np.array([
    [1.0, 0.0, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.0, 1.0],
  ])

  no.df.transition(model, c, t, df, "category")
  assert 2 not in df.category.value_counts()
  for i in [1,3]:
    assert df.category.value_counts()[i] > N/2 - sqrt(N) and df.category.value_counts()[i] < N/2 + sqrt(N)

  # spread evenly
  t = np.ones((3,3)) / 3  
  no.df.transition(model, c, t, df, "category")
  for i in c:
    assert df.category.value_counts()[i] > N/3 - sqrt(N) and df.category.value_counts()[i] < N/3 + sqrt(N)

  # all -> 1
  t = np.array([
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
  ])
  no.df.transition(model, c, t, df, "category")
  assert df.category.value_counts()[1] == N

def test():

  df = pd.read_csv("./tests/df.csv")

  # base model for MC engine
  model = no.Model(no.Timeline.null(), no.MonteCarlo.deterministic_identical_stream)

  cats = np.array(range(4)) 
  # identity matrix means no transitions
  trans = np.identity(len(cats))

  no.df.transition(model, cats, trans, df, "DC2101EW_C_ETHPUK11")

  assert len(df["DC2101EW_C_ETHPUK11"].unique()) == 1 and df["DC2101EW_C_ETHPUK11"].unique()[0] == 2

  # NOTE transition matrix interpreted as being COLUMN MAJOR due to pandas DataFrame storing data in column-major order

  # force 2->3
  trans[2,2] = 0.0
  trans[2,3] = 1.0
  no.df.transition(model, cats, trans, df, "DC2101EW_C_ETHPUK11")
  no.log(df["DC2101EW_C_ETHPUK11"].unique())
  assert len(df["DC2101EW_C_ETHPUK11"].unique()) == 1 and df["DC2101EW_C_ETHPUK11"].unique()[0] == 3


  # ~half of 3->0
  trans[3,0] = 0.5
  trans[3,3] = 0.5
  no.df.transition(model, cats, trans, df, "DC2101EW_C_ETHPUK11")
  assert np.array_equal(np.sort(df["DC2101EW_C_ETHPUK11"].unique()), np.array([0, 3]))


# def todo():

#   # define some global variables describing where the starting population and the parameters of the dynamics come from
#   initial_population = "examples/households/data/ssm_hh_E09000001_OA11_2011.csv"

#   hh = pd.read_csv(initial_population)

#   print(hh.columns.values)
#   c = hh.LC4408_C_AHTHUK11.unique()
#   print(c)
#   t = np.identity(len(c))

#   # [ 3  5  1  2 -1  4]
#   t = np.array([[0.9,  0.05, 0.05, 0.0,  0.0,  0.0],
#                 [0.05, 0.9,  0.04, 0.01, 0.0,  0.0],
#                 [0.0,  0.05, 0.9,  0.05, 0.0,  0.0],
#                 [0.0,  0.0,  0.05, 0.9,  0.05, 0.0],
#                 [0.1,  0.1,  0.1,  0.1,  0.5,  0.1],
#                 [0.0,  0.0,  0.00, 0.0,  0.2,  0.8]])

#   #print(t[1]) # horz
#   #print(t[:,1]) # vert
#   tc = np.cumsum(t, axis=1)
#   # TODO timing...
#   u = np.random.sample(len(hh))
#   for i in range(len(hh)):
#     current = hh.loc[i, "LC4408_C_AHTHUK11"]
#     hh.loc[i, "LC4408_C_AHTHUK11"] = sample(u[i], tc[current], c)

#   print(hh.LC4408_C_AHTHUK11.head())

#   tc = np.cumsum(t, axis=1)

#   print(np.cumsum(t[1]))
#   #print()