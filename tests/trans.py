
import numpy as np
import pandas as pd
import neworder as no

# def sample(u, t, c):
#   i = int(np.interp(u, t, range(len(t))))
#   return c[i]

import test as test_


def test():

  t = test_.Test()

  df = pd.read_csv("../../tests/df.csv")

  cats = np.array(range(4))
  # identity matrix means no transitions
  trans = np.identity(len(cats))
  no.transition(cats, trans, df, "DC2101EW_C_ETHPUK11")

  t.check(len(df["DC2101EW_C_ETHPUK11"].unique()) == 1 and df["DC2101EW_C_ETHPUK11"].unique()[0] == 2) 

  # NOTE transition matrix interpreted as being COLUMN MAJOR due to pandas DataFrame storing data in column-major order

  # force 2->3
  trans[2,2] = 0.0
  trans[3,2] = 1.0
  no.transition(cats, trans, df, "DC2101EW_C_ETHPUK11")
  t.check(len(df["DC2101EW_C_ETHPUK11"].unique()) == 1 and df["DC2101EW_C_ETHPUK11"].unique()[0] == 3) 

  # ~half of 3->0
  trans[0,3] = 0.5
  trans[3,3] = 0.5
  no.transition(cats, trans, df, "DC2101EW_C_ETHPUK11")
  t.check(np.array_equal(np.sort(df["DC2101EW_C_ETHPUK11"].unique()), np.array([0, 3]))) 

  return not t.any_failed

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