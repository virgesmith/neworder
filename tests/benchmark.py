import time
import numpy as np
import pandas as pd
import neworder as no

from math import sqrt


# define some global variables describing where the starting population and the parameters of the dynamics come from
INITIAL_POPULATION = "./tests/ssm_hh_E09000001_OA11_2011.csv"

t = np.array([ 
  [0.9,  0.05, 0.00, 0.0,  0.1,  0.0], 
  [0.05, 0.9,  0.05, 0.0,  0.1,  0.0], 
  [0.05, 0.04, 0.9,  0.05, 0.1,  0.0], 
  [0.0,  0.01, 0.05, 0.9,  0.1,  0.0], 
  [0.0,  0.0,  0.0,  0.05, 0.5,  0.2], 
  [0.0,  0.0,  0.00, 0.0,  0.1,  0.8]])


def sample(u, t, c):
  i = int(np.interp(u, t, range(len(t))))
  return c[i]

def python_impl(m):

  hh = pd.read_csv(INITIAL_POPULATION)
  # for i in range(3):
  #   hh = hh.append(hh, ignore_index=True)

  c = hh.LC4408_C_AHTHUK11.unique()

  start = time.time()

  u = m.mc().ustream(len(hh))
  tc = np.cumsum(t, axis=1)
  for i in range(len(hh)):
    current = hh.loc[i, "LC4408_C_AHTHUK11"]
    hh.loc[i, "LC4408_C_AHTHUK11"] = sample(u[i], tc[current], c)

  no.log(hh.LC4408_C_AHTHUK11)

  return len(hh), time.time() - start

def cpp_impl(m):

  hh = pd.read_csv(INITIAL_POPULATION)
  # for i in range(3):
  #   hh = hh.append(hh, ignore_index=True)

  #no.log(hh.head())
  no.log(hh.columns.values)
  c = hh.LC4408_C_AHTHUK11.unique()

  start = time.time()

  no.dataframe.transition(m, c, t, hh, "LC4408_C_AHTHUK11")

  no.log(hh.LC4408_C_AHTHUK11)

  return len(hh),  time.time() - start


def f(m):

  n = 1000

  c = [1,2,3]
  df = pd.DataFrame({"n": [1]*n})

  # no transitions
  t = np.identity(3)

  no.dataframe.transition(m, c, t, df, "n")
  no.log(df.n.value_counts()[1] == 1000)

  # all 1 -> 2
  t[0,0] = 0.0
  t[1,0] = 1.0
  no.dataframe.transition(m, c, t, df, "n")
  no.log(df.n.value_counts()[2] == 1000)

  # all 2 -> 1 or 3
  t = np.array([
    [1.0, 0.5, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 1.0],
  ])

  no.dataframe.transition(m, c, t, df, "n")
  no.log(2 not in df.n.value_counts())#[2] == 1000)
  no.log(df.n.value_counts())

  t = np.ones((3,3)) / 3  
  no.dataframe.transition(m, c, t, df, "n")
  no.log(df.n.value_counts())
  for i in c:
    no.log(df.n.value_counts()[i] > n/3 - sqrt(n) and df.n.value_counts()[i] < n/3 + sqrt(n))

  t = np.array([
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
  ])
  no.dataframe.transition(m, c, t, df, "n")
  no.log(df.n.value_counts())

if __name__ == "__main__":
  m = no.Model(no.Timeline.null(), no.MonteCarlo.deterministic_identical_stream)

  # rows, tc = cpp_impl(m)
  # no.log("C++ %d: %f" % (rows, tc))

  # m.mc().reset()
  # rows, tp = python_impl(m)
  # no.log("py  %d: %f" % (rows, tp))

  # no.log("speedup factor = %f" % (tp / tc))

  f(m)