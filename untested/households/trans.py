import time
import numpy as np
import pandas as pd

# timing comparison between C++ and python implementations:
#           TIME(s)
# ROWS   python C++     SPEEDUP
#   5608   2.1  0.00015  14000
#  50895  25.4  0.00152  16000
# 121672  84.5  0.00362  23000


def sample(u, t, c):
  i = int(np.interp(u, t, range(len(t))))
  return c[i]

def python_impl():

  # define some global variables describing where the starting population and the parameters of the dynamics come from
  initial_population = "examples/households/data/ssm_hh_E08000021_OA11_2011.csv"

  hh = pd.read_csv(initial_population)

  #print(hh.columns.values)
  c = hh.LC4408_C_AHTHUK11.unique()
  print(c)

  # [ 3  5  1  2 -1  4]
  t = np.array([ 
    [0.9,  0.05, 0.05, 0.0,  0.0,  0.0], 
    [0.05, 0.9,  0.04, 0.01, 0.0,  0.0], 
    [0.0,  0.05, 0.9,  0.05, 0.0,  0.0], 
    [0.0,  0.0,  0.05, 0.9,  0.05, 0.0], 
    [0.1,  0.1,  0.1,  0.1,  0.5,  0.1], 
    [0.0,  0.0,  0.00, 0.0,  0.2,  0.8]])

  #print(t[1]) # horz
  #print(t[:,1]) # vert
  # TODO timing...
  start = time.time()

  u = np.random.sample(len(hh))
  tc = np.cumsum(t, axis=1)
  for i in range(len(hh)):
    current = hh.loc[i, "LC4408_C_AHTHUK11"]
    hh.loc[i, "LC4408_C_AHTHUK11"] = sample(u[i], tc[current], c)

  print("%d: %f" % (len(hh),  time.time() - start))

  #print(hh.LC4408_C_AHTHUK11.head())

def cpp_impl():
  # define some global variables describing where the starting population and the parameters of the dynamics come from
  initial_population = "examples/households/data/ssm_hh_E08000021_OA11_2011.csv"

  hh = pd.read_csv(initial_population)

  #print(hh.columns.values)
  c = hh.LC4408_C_AHTHUK11.unique()
  print(c)

  # [ 3  5  1  2 -1  4]
  t = np.array([ 
    [0.9,  0.05, 0.05, 0.0,  0.0,  0.0], 
    [0.05, 0.9,  0.04, 0.01, 0.0,  0.0], 
    [0.0,  0.05, 0.9,  0.05, 0.0,  0.0], 
    [0.0,  0.0,  0.05, 0.9,  0.05, 0.0], 
    [0.1,  0.1,  0.1,  0.1,  0.5,  0.1], 
    [0.0,  0.0,  0.00, 0.0,  0.2,  0.8]])

  start = time.time()

  no.df.transition(c, t, hh, "LC4408_C_AHTHUK11")

  print("%d: %f" % (len(hh),  time.time() - start))

  #print(hh.LC4408_C_AHTHUK11.head())
